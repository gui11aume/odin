"""Build webdataset shards from the cleaned cluster corpus.

Each shard sample is one cluster (the collator-ready record):

    {"__key__": "cluster-000000000", "json": '{"tags": [...], "cells": [...]}'}

Two phases:

* Phase A (single process): one counting pass, then one streaming pass that
  writes a flat plain-text file of the train lines (recording the byte
  offset of every shard boundary), a val flat file, and the per-script
  letter-frequency table used by the collator's letter corruption.
* Phase B (N worker processes): each worker reads its contiguous shard
  range directly from the flat file (byte-offset seeks, no shared state)
  and writes its ``shard-NNNNNN.tar.gz`` files, re-opening each shard to
  verify the member count.

Usage:
    python runners/build_odin_shards.py --input /mnt/nvme1/odin_train_set.clean.txt.gz \
        --output /mnt/nvme1/odin_wds --workers 8 [--limit N] [--shard-size 2048] [--val-size 4000]
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import shutil
import sys
import tarfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import webdataset as wds

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_clusters_phase_3 import TAGS  # noqa: E402

log = logging.getLogger(__name__)

CORPUSE_ALPHA_SCRIPTS = frozenset(TAGS) - {"cn", "jp", "kr"}


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_line(line: str) -> tuple[list[str], list[str]] | None:
    """Parse one ``tag{value}`` tab-separated line; None when malformed."""
    line = line.rstrip("\r\n")
    tags: list[str] = []
    cells: list[str] = []
    for cell in line.split("\t"):
        if len(cell) < 4 or cell[2] != "{" or not cell.endswith("}"):
            return None
        tag = cell[:2]
        if tag not in TAGS:
            return None
        value = cell[3:-1]
        if value == "":
            return None
        tags.append(tag)
        cells.append(value)
    if "la" not in tags:
        return None
    return tags, cells


def record_json(tags: list[str], cells: list[str]) -> bytes:
    return json.dumps({"tags": tags, "cells": cells}, ensure_ascii=False).encode("utf-8")


# --------------------------------------------------------------------------- #
# Phase A
# --------------------------------------------------------------------------- #
def iter_lines(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        yield from handle


def count_lines(path: Path, limit: int = 0) -> int:
    total = 0
    for _ in iter_lines(path):
        total += 1
        if limit and total >= limit:
            break
    return total


def val_index_set(total: int, val_size: int) -> set[int]:
    """Deterministic stratified val selection, evenly spread over the corpus.

    The k-th val slot is line ``(k * total) // val_size`` for k in 0..val_size-1,
    so val lines are interleaved at a regular cadence and the split is
    reproducible across runs and re-builds.
    """
    if total <= 0 or val_size <= 0:
        return set()
    return {(k * total) // val_size for k in range(val_size)}


def phase_a(
    input_path: Path,
    work_dir: Path,
    shard_size: int,
    val_size: int,
    limit: int = 0,
    val_input: Path | None = None,
    test_input: Path | None = None,
) -> dict:
    """Count, split train/val(/test), write the flat files, offsets, and letter frequencies.

    With ``val_input`` given, those lines are the val set verbatim (file order,
    deduped) and the stratified selection is skipped; any val line that also
    appears in the input is skipped there (leakage guard). Otherwise val is
    drawn from the input by the deterministic stratified scheme. ``test_input``,
    if given, is the test set verbatim and is skipped in the input as well.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    flat_path = work_dir / "flat_train.txt"
    val_path = work_dir / "flat_val.txt"
    offsets_path = work_dir / "offsets.json"
    freq_path = work_dir / "char_frequencies.json"

    log.info("Phase A.1: counting lines ...")
    t0 = time.time()
    total = count_lines(input_path, limit)
    log.info("Phase A.1: %d lines in %.1fs", total, time.time() - t0)

    val_lines: list[str] = []
    val_set: set[str] = set()
    if val_input is not None:
        for line in iter_lines(val_input):
            text = line.rstrip("\r\n")
            if text and text not in val_set:
                val_set.add(text)
                val_lines.append(text)
        n_val = len(val_lines)
        n_train = total  # adjusted below if any val lines appear in the input
        val_indices = None
        log.info("Phase A: %d explicit val lines from %s (stratified selection skipped)", n_val, val_input)
        with open(val_path, "wb") as val:
            for text in val_lines:
                val.write((text + "\n").encode("utf-8"))
    else:
        n_val = min(val_size, total // 2)
        n_train = total - n_val
        val_indices = val_index_set(total, n_val)

    test_lines: list[str] = []
    test_set: set[str] = set()
    if test_input is not None:
        for line in iter_lines(test_input):
            text = line.rstrip("\r\n")
            if text and text not in test_set:
                test_set.add(text)
                test_lines.append(text)
        log.info("Phase A: %d explicit test lines from %s", len(test_lines), test_input)
        with open(work_dir / "flat_test.txt", "wb") as test:
            for text in test_lines:
                test.write((text + "\n").encode("utf-8"))
    n_shards = (n_train + shard_size - 1) // shard_size

    log.info(
        "Phase A.2: writing flat files (n_train=%d, n_val=%d, n_test=%d, n_shards=%d) ...",
        n_train,
        n_val,
        len(test_lines),
        n_shards,
    )
    t0 = time.time()
    freq: Counter = Counter()
    malformed = 0
    offsets: list[int] = []
    train_line_idx = 0
    val_line_idx = 0
    val_overlap = 0
    test_overlap = 0
    # The val file is written from the stream only for the stratified split; for an
    # explicit --val-input it was already written above and must not be
    # re-opened (that would truncate it), so a throwaway sink is used.
    with ExitStack() as stack:
        flat = stack.enter_context(open(flat_path, "wb"))
        val = stack.enter_context(open(val_path, "wb") if val_indices is not None else io.BytesIO())
        for i, line in enumerate(iter_lines(input_path)):
            if limit and i >= limit:
                break
            parsed = parse_line(line)
            if parsed is None:
                malformed += 1
                continue
            tags, cells = parsed
            for tag, cell in zip(tags, cells):
                if tag in CORPUSE_ALPHA_SCRIPTS:
                    freq.update((f"{tag}|{ch}" for ch in cell if ch.isalpha()))
            data = line if line.endswith("\n") else line + "\n"
            if val_indices is not None and i in val_indices:
                val.write(data.encode("utf-8"))
                val_line_idx += 1
                continue
            if val_set and line.rstrip("\n") in val_set:
                val_overlap += 1
                continue  # val line present in the train input: never train on it
            if test_set and line.rstrip("\n") in test_set:
                test_overlap += 1
                continue  # test line present in the train input: never train on it
            if train_line_idx % shard_size == 0:
                offsets.append(flat.tell())
            flat.write(data.encode("utf-8"))
            train_line_idx += 1
        offsets.append(flat.tell())
    # Derive the actual sizes from the offset bookkeeping (robust if a few
    # malformed lines slipped into the input: they are skipped, not sharded).
    if train_line_idx != n_train:
        log.warning(
            "Phase A: %d lines skipped as malformed; actual n_train=%d (predicted %d).",
            n_train - train_line_idx,
            train_line_idx,
            n_train,
        )
        n_train = train_line_idx
        n_shards = (n_train + shard_size - 1) // shard_size
    if len(offsets) != n_shards + 1:
        raise RuntimeError(f"Offset bookkeeping mismatch: {len(offsets) - 1} boundaries for {n_shards} shards.")

    with open(offsets_path, "w", encoding="utf-8") as fh:
        json.dump(offsets, fh)
    letter_freqs: dict[str, list[list]] = {}
    for script in sorted(TAGS):
        pairs = [
            (ch, count) for key, count in freq.items() if key.startswith(f"{script}|") for ch in [key.split("|", 1)[1]]
        ]
        if pairs:
            letter_freqs[script] = sorted(pairs, key=lambda p: (-p[1], p[0]))
    with open(freq_path, "w", encoding="utf-8") as fh:
        json.dump(letter_freqs, fh, ensure_ascii=False)

    if val_overlap:
        log.warning("Phase A: %d val lines also appeared in the train input and were skipped.", val_overlap)
    if test_overlap:
        log.warning("Phase A: %d test lines also appeared in the train input and were skipped.", test_overlap)
    stats = {
        "total": total,
        "n_train": n_train,
        "n_val": val_line_idx if val_indices is not None else n_val,
        "n_test": len(test_lines),
        "n_shards": n_shards,
        "malformed": malformed,
        "val_overlap": val_overlap,
        "test_overlap": test_overlap,
        "phase_a_seconds": round(time.time() - t0, 1),
    }
    log.info("Phase A done: %s (%.1fs)", stats, stats["phase_a_seconds"])
    if malformed:
        log.warning("Phase A: %d malformed lines skipped (input should already be clean).", malformed)
    return stats


# --------------------------------------------------------------------------- #
# Phase B
# --------------------------------------------------------------------------- #
def build_shard_range(
    flat_path: str,
    offsets: list[int],
    shard_lo: int,
    shard_hi: int,
    out_dir: str,
    shard_size: int,
    n_train: int,
) -> list[dict]:
    """Build the shards in [shard_lo, shard_hi) from the flat file. Worker entry point."""
    results: list[dict] = []
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(flat_path, "rb") as flat:
        for s in range(shard_lo, shard_hi):
            lo, hi = offsets[s], offsets[s + 1]
            flat.seek(lo)
            blob = flat.read(hi - lo)
            n_expected = min(shard_size, n_train - s * shard_size)
            lines = [ln for ln in blob.decode("utf-8").splitlines() if ln]
            if len(lines) != n_expected:
                raise RuntimeError(f"Shard {s}: expected {n_expected} lines, found {len(lines)}.")
            name = f"shard-{s:06d}.tar.gz"
            sink = wds.TarWriter(str(out / name))
            try:
                for j, line in enumerate(lines):
                    parsed = parse_line(line)
                    if parsed is None:
                        raise RuntimeError(f"Malformed line in shard {s} offset {j} (input should be clean).")
                    tags, cells = parsed
                    sink.write({"__key__": f"cluster-{s * shard_size + j:09d}", "json": record_json(tags, cells)})
            finally:
                sink.close()
            # Verify: re-open the shard and count members.
            with tarfile.open(str(out / name), "r:gz") as tf:
                n_members = sum(1 for _ in tf)
            if n_members != n_expected:
                raise RuntimeError(f"Shard {name}: {n_members} members, expected {n_expected}.")
            results.append({"shard": name, "n_clusters": n_expected})
    return results


def build_val_shards(val_path: Path, out_dir: Path, shard_size: int, key_prefix: str = "val-") -> list[dict]:
    """Write the (small) val/test shards. Run in the main process."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines = [ln for ln in val_path.read_text(encoding="utf-8").splitlines() if ln]
    results = []
    for s in range(0, len(lines), shard_size):
        chunk = lines[s : s + shard_size]
        name = f"shard-{s // shard_size:06d}.tar.gz"
        sink = wds.TarWriter(str(out / name))
        try:
            for j, line in enumerate(chunk):
                parsed = parse_line(line)
                if parsed is None:
                    raise RuntimeError(
                        f"Malformed {key_prefix.rstrip('-')} line at offset {s + j} (input should be clean)."
                    )
                tags, cells = parsed
                # Namespaced key: indices restart at 0 and must not collide with train keys.
                sink.write({"__key__": f"{key_prefix}{s + j:09d}", "json": record_json(tags, cells)})
        finally:
            sink.close()
        results.append({"shard": name, "n_clusters": len(chunk)})
    return results


def phase_b(
    work_dir: Path, train_dir: Path, val_dir: Path, test_dir: Path, shard_size: int, workers: int, stats: dict
) -> dict:
    offsets = json.loads((work_dir / "offsets.json").read_text(encoding="utf-8"))
    n_shards = len(offsets) - 1
    n_train = stats["n_train"]
    bounds = [round(w * n_shards / workers) for w in range(workers + 1)]
    log.info("Phase B: %d shards across %d workers: %s", n_shards, workers, bounds)
    t0 = time.time()
    train_results: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = []
        for w in range(workers):
            lo, hi = bounds[w], bounds[w + 1]
            if lo >= hi:
                continue
            futures.append(
                pool.submit(
                    build_shard_range,
                    str(work_dir / "flat_train.txt"),
                    offsets,
                    lo,
                    hi,
                    str(train_dir),
                    shard_size,
                    n_train,
                )
            )
            log.info("Worker %d: shards [%d, %d)", w, lo, hi)
        for future in futures:
            train_results += future.result()
    val_results = build_val_shards(work_dir / "flat_val.txt", val_dir, shard_size)
    test_results = (
        build_val_shards(work_dir / "flat_test.txt", test_dir, shard_size, key_prefix="test-")
        if (work_dir / "flat_test.txt").exists()
        else []
    )
    stats["phase_b_seconds"] = round(time.time() - t0, 1)
    return {"train": train_results, "val": val_results, "test": test_results}


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    parser = argparse.ArgumentParser(description="Build webdataset shards from the cleaned cluster corpus.")
    parser.add_argument("--input", required=True, help="Cleaned cluster corpus (.txt or .txt.gz).")
    parser.add_argument("--output", required=True, help="Output root (train/ and val/ shard dirs).")
    parser.add_argument("--shard-size", type=int, default=2048, help="Clusters per shard (default 2048).")
    parser.add_argument("--val-size", type=int, default=4000, help="Number of val clusters (default 4000).")
    parser.add_argument("--workers", type=int, default=8, help="Phase B worker processes (default 8).")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N clusters (0 = all).")
    parser.add_argument(
        "--val-input",
        default=None,
        help="Explicit val corpus (.txt/.txt.gz): used verbatim as the val set (file order), "
        "and its lines are skipped in the train input; the stratified --val-size selection is skipped.",
    )
    parser.add_argument(
        "--test-input",
        default=None,
        help="Explicit test corpus (.txt/.txt.gz): used verbatim as the test set (file order), "
        "and its lines are skipped in the train input.",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_root = Path(args.output)
    if not input_path.is_file():
        raise SystemExit(f"Input not found: {args.input}")
    train_dir = output_root / "train"
    val_dir = output_root / "val"
    test_dir = output_root / "test"
    work_dir = output_root / "work"
    for d in (train_dir, val_dir, test_dir, work_dir):
        d.mkdir(parents=True, exist_ok=True)

    val_input_path = None
    if args.val_input is not None:
        val_input_path = Path(args.val_input)
        if not val_input_path.is_file():
            raise SystemExit(f"Val input not found: {args.val_input}")
        if args.val_size != 4000:
            log.warning("--val-input given: --val-size %d is ignored (explicit val lines win).", args.val_size)
    test_input_path = None
    if args.test_input is not None:
        test_input_path = Path(args.test_input)
        if not test_input_path.is_file():
            raise SystemExit(f"Test input not found: {args.test_input}")

    t_start = time.time()
    stats = phase_a(
        input_path,
        work_dir,
        args.shard_size,
        args.val_size,
        args.limit,
        val_input=val_input_path,
        test_input=test_input_path,
    )
    shard_results = phase_b(work_dir, train_dir, val_dir, test_dir, args.shard_size, args.workers, stats)

    n_train_shards = len(shard_results["train"])
    n_val_shards = len(shard_results["val"])
    n_test_shards = len(shard_results["test"])
    manifest = {
        "input": str(input_path),
        "val_input": str(val_input_path) if val_input_path else None,
        "test_input": str(test_input_path) if test_input_path else None,
        "shard_size": args.shard_size,
        "n_total": stats["total"],
        "n_train": stats["n_train"],
        "n_val": stats["n_val"],
        "n_test": stats["n_test"],
        "n_train_shards": n_train_shards,
        "n_val_shards": n_val_shards,
        "n_test_shards": n_test_shards,
        "workers": args.workers,
        "total_seconds": round(time.time() - t_start, 1),
        "train_pattern": f"train/shard-{{000000..{n_train_shards - 1:06d}}}.tar.gz" if n_train_shards else None,
        "val_pattern": f"val/shard-{{000000..{n_val_shards - 1:06d}}}.tar.gz" if n_val_shards else None,
        "test_pattern": f"test/shard-{{000000..{n_test_shards - 1:06d}}}.tar.gz" if n_test_shards else None,
    }
    with open(output_root / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    log.info("Build complete in %.1fs:", manifest["total_seconds"])
    log.info("  train: %d clusters in %d shards  (%s)", manifest["n_train"], n_train_shards, manifest["train_pattern"])
    log.info("  val:   %d clusters in %d shards  (%s)", manifest["n_val"], n_val_shards, manifest["val_pattern"])
    log.info("  test:  %d clusters in %d shards  (%s)", manifest["n_test"], n_test_shards, manifest["test_pattern"])
    log.info("  output root: %s", output_root)
    # Promote the letter-frequency table, remove the large intermediates.
    shutil.copyfile(work_dir / "char_frequencies.json", output_root / "char_frequencies.json")
    (work_dir / "flat_train.txt").unlink(missing_ok=True)
    (work_dir / "flat_val.txt").unlink(missing_ok=True)
    (work_dir / "flat_test.txt").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
