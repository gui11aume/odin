"""Merge the inventor and company cluster corpora into one training pool.

The company clusters (runners/extract_companies.py +
runners/generate_companies_phase_1.py) use the same 12 tags and the same
``tag{value}`` line format as the inventor clusters, so a single Odin model
trains on both. This runner produces:

* the merged train pool (inventor lines in original order, then company
  lines in original order), and
* the new val corpus: the previous val lines verbatim (byte-identical, so
  val stays comparable to the pre-merge runs) followed by a fixed-seed
  holdout of company lines.

Rules applied:

* every cell whose tag token + surface exceeds ``--max-tokens`` is dropped
  (the line is kept with its remaining cells). The collator would otherwise
  hard-truncate such a surface at ``max_surface_tokens``, which would teach
  the model a name cut off mid-word;
* lines from ``--old-val`` are excluded from the pool (they are val, not
  train), and the company holdout is excluded as well;
* ordering and line content are otherwise preserved.

Usage:
    python runners/merge_odin_trainset.py INVENTOR COMPANIES \
        --old-val OLD_VAL.txt.gz -o POOL.txt.gz --val-out VAL.txt.gz \
        [--holdout 2000] [--seed 123] [--max-tokens 48]
"""

from __future__ import annotations

import argparse
import gzip
import io
import logging
import random
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_odin_shards import parse_line  # noqa: E402

log = logging.getLogger(__name__)


def iter_lines(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        yield from handle


def open_out_gz(path: Path):
    return io.TextIOWrapper(gzip.GzipFile(path, mode="wb", mtime=0), encoding="utf-8")


def format_cells(cells: list[tuple[str, str]]) -> str:
    return "\t".join(f"{tag}{{{value}}}" for tag, value in cells)


def drop_overlong_cells(cells: list[tuple[str, str]], tokenizer, max_tokens: int) -> tuple[list[tuple[str, str]], int]:
    """Drop cells whose tag token + surface exceeds max_tokens (collator rule)."""
    kept: list[tuple[str, str]] = []
    dropped = 0
    for tag, value in cells:
        if 1 + len(tokenizer.encode(value, add_special_tokens=False)) > max_tokens:
            dropped += 1
            continue
        kept.append((tag, value))
    return kept, dropped


def pick_holdout(n_lines: int, n_holdout: int, seed: int) -> set[int]:
    """Deterministic holdout indices (stable for a fixed seed and line count)."""
    n_holdout = min(n_holdout, n_lines)
    if n_holdout == 0:
        return set()
    return set(random.Random(seed).sample(range(n_lines), n_holdout))


def merge(
    inventor: Path,
    companies: Path,
    old_val: Path,
    pool_out: Path,
    val_out: Path,
    *,
    holdout: int,
    seed: int,
    tokenizer,
    max_tokens: int,
) -> dict:
    val_lines: list[str] = []
    seen_val: set[str] = set()
    for line in iter_lines(old_val):
        text = line.rstrip("\r\n")
        if text and text not in seen_val:
            seen_val.add(text)
            val_lines.append(text)
    log.info("old val lines: %d", len(val_lines))

    # --- Companies (held in memory: ~225k lines): prune cells, then split ---
    company_lines = [ln.rstrip("\r\n") for ln in iter_lines(companies)]
    holdout_idx = pick_holdout(len(company_lines), holdout, seed)
    log.info("company lines: %d, holdout: %d (seed %d)", len(company_lines), len(holdout_idx), seed)

    pruned: list[list[tuple[str, str]] | None] = []
    company_dropped = 0
    company_empty = 0
    for line in company_lines:
        parsed = parse_line(line)
        if parsed is None:
            raise RuntimeError(f"Malformed company line (input should be phase-3 clean): {line[:120]!r}")
        cells, dropped = drop_overlong_cells(list(zip(parsed[0], parsed[1])), tokenizer, max_tokens)
        company_dropped += dropped
        if not cells:
            company_empty += 1
        pruned.append(cells if cells else None)

    val_holdout: list[str] = []
    pool_companies: list[str] = []
    for i, cells in enumerate(pruned):
        if cells is None:
            continue
        text = format_cells(cells)
        if i in holdout_idx:
            val_holdout.append(text)
        else:
            pool_companies.append(text)
    log.info(
        "company pruned cells: %d, empty lines dropped: %d, holdout lines: %d",
        company_dropped,
        company_empty,
        len(val_holdout),
    )

    # --- New val corpus: old val verbatim, then the company holdout ---
    with open_out_gz(val_out) as val_fh:
        for text in val_lines:
            print(text, file=val_fh)
        for text in val_holdout:
            print(text, file=val_fh)

    # --- Stream the inventor set: skip old-val lines, prune cells, write pool ---
    inventor_dropped = 0
    inventor_kept = 0
    inventor_skipped_val = 0
    with open_out_gz(pool_out) as pool_fh:
        for line in iter_lines(inventor):
            text = line.rstrip("\r\n")
            if text in seen_val:
                inventor_skipped_val += 1
                continue
            parsed = parse_line(text)
            if parsed is None:
                raise RuntimeError(f"Malformed inventor line (input should be phase-3 clean): {text[:120]!r}")
            cells, dropped = drop_overlong_cells(list(zip(parsed[0], parsed[1])), tokenizer, max_tokens)
            if not cells:
                log.warning("Dropped inventor line left empty after pruning: %r", text[:120])
                continue
            inventor_dropped += dropped
            print(format_cells(cells), file=pool_fh)
            inventor_kept += 1
        for text in pool_companies:
            print(text, file=pool_fh)

    return {
        "old_val": len(val_lines),
        "company_lines": len(company_lines),
        "company_pruned_cells": company_dropped,
        "company_empty_dropped": company_empty,
        "company_holdout": len(val_holdout),
        "company_to_pool": len(pool_companies),
        "inventor_skipped_val": inventor_skipped_val,
        "inventor_pruned_cells": inventor_dropped,
        "inventor_to_pool": inventor_kept,
        "pool_total": inventor_kept + len(pool_companies),
        "val_total": len(val_lines) + len(val_holdout),
    }


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    parser = argparse.ArgumentParser(description="Merge inventor + company cluster corpora into one train pool.")
    parser.add_argument("inventor", help="Inventor cluster corpus (.txt or .txt.gz), phase-3 clean.")
    parser.add_argument("companies", help="Company cluster corpus (.txt or .txt.gz), phase-3 clean.")
    parser.add_argument("--old-val", required=True, help="Previous val corpus (kept as val, never trained on).")
    parser.add_argument("-o", "--pool", required=True, help="Output merged train pool (.txt.gz).")
    parser.add_argument("--val-out", required=True, help="Output new val corpus: old val + company holdout (.txt.gz).")
    parser.add_argument("--holdout", type=int, default=2000, help="Company lines held out into val (default 2000).")
    parser.add_argument("--seed", type=int, default=123, help="Holdout selection seed (default 123).")
    parser.add_argument("--tokenizer", default="/mnt/nvme1/odin_tokenizer", help="Byte-BPE tokenizer directory.")
    parser.add_argument("--max-tokens", type=int, default=48, help="Collator max_surface_tokens (default 48).")
    args = parser.parse_args(argv)

    for p in (args.inventor, args.companies, args.old_val):
        if not Path(p).is_file():
            raise SystemExit(f"Input not found: {p}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)  # nosec: B615  # local directory, not the Hub
    stats = merge(
        Path(args.inventor),
        Path(args.companies),
        Path(args.old_val),
        Path(args.pool),
        Path(args.val_out),
        holdout=args.holdout,
        seed=args.seed,
        tokenizer=tokenizer,
        max_tokens=args.max_tokens,
    )
    for key, value in stats.items():
        log.info("%-24s %s", key, value)


if __name__ == "__main__":
    main()
