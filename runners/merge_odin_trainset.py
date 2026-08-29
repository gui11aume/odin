"""Merge the inventor and company cluster corpora into one training pool.

The company clusters (runners/extract_companies.py +
runners/generate_companies_phase_1.py) use the same 12 tags and the same
``tag{value}`` line format as the inventor clusters, so a single Odin model
trains on both. This runner produces:

* the merged train pool (inventor lines in original order, then company
  lines in original order),
* the new val corpus: the previous val lines verbatim (byte-identical, so
  val stays comparable to the pre-merge runs) followed by a fixed-seed
  holdout of company lines, and optionally
* a test corpus: a fixed-seed carve-out of inventor and company lines,
  disjoint from both the pool and val (never trained on, never used for
  selection).

Rules applied:

* every cell whose tag token + surface exceeds ``--max-tokens`` is dropped
  (the line is kept with its remaining cells). The collator would otherwise
  hard-truncate such a surface at ``max_surface_tokens``, which would teach
  the model a name cut off mid-word;
* lines from ``--old-val`` are excluded from the pool (they are val, not
  train), and the company holdout is excluded as well;
* test lines are drawn only from lines that would otherwise enter the pool
  (never from val, and with a different seed), so all three sets are
  disjoint by construction;
* ordering and line content are otherwise preserved.

Usage:
    python runners/merge_odin_trainset.py INVENTOR COMPANIES \
        --old-val OLD_VAL.txt.gz -o POOL.txt.gz --val-out VAL.txt.gz \
        [--holdout 2000] [--seed 123] [--max-tokens 48] \
        [--test-out TEST.txt.gz --test-inv 2000 --test-comp 1000 --test-seed 456]
"""

from __future__ import annotations

import argparse
import gzip
import io
import logging
import random
import sys
from contextlib import ExitStack
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
    """Drop cells whose tag token + surface exceeds max_tokens (collator rule).

    Fast path: a byte-BPE token is at least one byte, so a surface of at most
    max_tokens-1 UTF-8 bytes cannot exceed the limit without tokenizing. Only
    the remaining cells are batch-tokenized.
    """
    limit_bytes = max_tokens - 1
    flags: list[bool | None] = []
    pending: list[tuple[int, str, str]] = []  # (index, tag, value)
    for tag, value in cells:
        if len(value.encode("utf-8")) <= limit_bytes:
            flags.append(True)
        else:
            pending.append((len(flags), tag, value))
            flags.append(None)
    if pending:
        encodings = tokenizer.encode([v for _i, _t, v in pending], add_special_tokens=False)
        for (i, _t, _v), ids in zip(pending, encodings):
            flags[i] = 1 + len(ids) <= max_tokens
    kept: list[tuple[str, str]] = []
    dropped = 0
    for ok, (tag, value) in zip(flags, cells):
        if ok:
            kept.append((tag, value))
        else:
            dropped += 1
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
    test_out: Path | None = None,
    test_inv: int = 0,
    test_comp: int = 0,
    test_seed: int = 456,
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
    company_dead = 0
    for line in company_lines:
        parsed = parse_line(line)
        if parsed is None:
            raise RuntimeError(f"Malformed company line (input should be phase-3 clean): {line[:120]!r}")
        cells, dropped = drop_overlong_cells(list(zip(parsed[0], parsed[1])), tokenizer, max_tokens)
        company_dropped += dropped
        # A line whose only overlong cells were its la cells is useless: the
        # shard builder requires at least one la cell, and a truncated one
        # would be a cut-off label.
        if not cells or not any(tag == "la" for tag, _ in cells):
            company_dead += 1
        pruned.append(cells if cells and any(tag == "la" for tag, _ in cells) else None)

    # Test carve-out: drawn only from the company lines that would otherwise
    # enter the pool, so it is disjoint from the val holdout by construction.
    comp_test_set: set[int] = set()
    if test_out is not None and test_comp > 0:
        pool_idx = [i for i, cells in enumerate(pruned) if cells is not None and i not in holdout_idx]
        comp_test_set = {
            pool_idx[j] for j in random.Random(test_seed).sample(range(len(pool_idx)), min(test_comp, len(pool_idx)))
        }
        log.info("company test carve-out: %d (seed %d)", len(comp_test_set), test_seed)

    val_holdout: list[str] = []
    test_companies: list[str] = []
    pool_companies: list[str] = []
    for i, cells in enumerate(pruned):
        if cells is None:
            continue
        text = format_cells(cells)
        if i in holdout_idx:
            val_holdout.append(text)
        elif i in comp_test_set:
            test_companies.append(text)
        else:
            pool_companies.append(text)
    log.info(
        "company pruned cells: %d, dead lines dropped (no la left): %d, holdout lines: %d, test lines: %d",
        company_dropped,
        company_dead,
        len(val_holdout),
        len(test_companies),
    )

    # --- New val corpus: old val verbatim, then the company holdout ---
    with open_out_gz(val_out) as val_fh:
        for text in val_lines:
            print(text, file=val_fh)
        for text in val_holdout:
            print(text, file=val_fh)

    # --- Inventor test carve-out (needs the eligible line count first) ---
    inv_test_set: set[int] = set()
    if test_out is not None and test_inv > 0:
        n_eligible = 0
        for line in iter_lines(inventor):
            if line.rstrip("\r\n") not in seen_val:
                n_eligible += 1
        inv_test_set = set(random.Random(test_seed).sample(range(n_eligible), min(test_inv, n_eligible)))
        log.info("inventor test carve-out: %d of %d eligible lines (seed %d)", len(inv_test_set), n_eligible, test_seed)

    # --- Stream the inventor set: skip old-val lines, prune cells, write pool/test ---
    inventor_dropped = 0
    inventor_kept = 0
    inventor_to_test = 0
    inventor_skipped_val = 0
    inventor_dead = 0
    with ExitStack() as stack:
        pool_fh = stack.enter_context(open_out_gz(pool_out))
        test_fh = stack.enter_context(open_out_gz(test_out)) if test_out is not None else None
        elig = 0
        for line in iter_lines(inventor):
            text = line.rstrip("\r\n")
            if text in seen_val:
                inventor_skipped_val += 1
                continue
            parsed = parse_line(text)
            if parsed is None:
                raise RuntimeError(f"Malformed inventor line (input should be phase-3 clean): {text[:120]!r}")
            cells, dropped = drop_overlong_cells(list(zip(parsed[0], parsed[1])), tokenizer, max_tokens)
            inventor_dropped += dropped
            if not cells or not any(tag == "la" for tag, _ in cells):
                inventor_dead += 1
                log.warning("Dropped inventor line left without an la cell after pruning: %r", text[:120])
                elig += 1
                continue
            out_text = format_cells(cells)
            if test_fh is not None and elig in inv_test_set:
                print(out_text, file=test_fh)
                inventor_to_test += 1
            else:
                print(out_text, file=pool_fh)
                inventor_kept += 1
            elig += 1
        if test_fh is not None:
            for text in test_companies:
                print(text, file=test_fh)
        for text in pool_companies:
            print(text, file=pool_fh)

    return {
        "old_val": len(val_lines),
        "company_lines": len(company_lines),
        "company_pruned_cells": company_dropped,
        "company_dead_dropped": company_dead,
        "company_holdout": len(val_holdout),
        "company_to_pool": len(pool_companies),
        "inventor_skipped_val": inventor_skipped_val,
        "inventor_pruned_cells": inventor_dropped,
        "inventor_dead_dropped": inventor_dead,
        "inventor_to_pool": inventor_kept,
        "pool_total": inventor_kept + len(pool_companies),
        "val_total": len(val_lines) + len(val_holdout),
        "test_inventor": inventor_to_test,
        "test_company": len(test_companies),
        "test_total": inventor_to_test + len(test_companies),
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
    parser.add_argument(
        "--test-out", default=None, help="Output test corpus (.txt.gz); also enables the test carve-out."
    )
    parser.add_argument(
        "--test-inv", type=int, default=2000, help="Inventor lines carved out into test (default 2000)."
    )
    parser.add_argument(
        "--test-comp", type=int, default=1000, help="Company lines carved out into test (default 1000)."
    )
    parser.add_argument(
        "--test-seed", type=int, default=456, help="Test selection seed, distinct from --seed (default 456)."
    )
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
        test_out=Path(args.test_out) if args.test_out else None,
        test_inv=args.test_inv,
        test_comp=args.test_comp,
        test_seed=args.test_seed,
    )
    for key, value in stats.items():
        log.info("%-24s %s", key, value)


if __name__ == "__main__":
    main()
