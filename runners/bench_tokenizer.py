"""Benchmark the C tokenizer against HuggingFace.

Streams cells from a WebDataset split and measures encode throughput for:

  * HF              – PreTrainedTokenizerFast.encode (the baseline to beat)
  * C-encode        – one C call per string (encode)
  * C-encode_many-1 – batched, 1 thread (isolates the GIL-release + batch
                      overhead from parallelism)
  * C-encode_many-N – batched, N threads (the multi-CPU win)
  * C-encode_padded – batched encode + right-pad (the collator's shape)

Usage:
    python runners/bench_tokenizer.py [--split test] [--limit 200000] [--threads 8]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WDS_ROOT = Path("/mnt/nvme1/odin_wds_v3")
TOKENIZER_JSON = Path("/mnt/nvme1/odin_tokenizer/tokenizer.json")


def load_cells(split: str, limit: int) -> list[str]:
    import json
    import tarfile

    cells: list[str] = []
    for shard in sorted((WDS_ROOT / split).glob("*.tar.gz")):
        with tarfile.open(shard) as tf:
            for m in tf.getmembers():
                if not m.isfile() or not m.name.endswith(".json"):
                    continue
                obj = json.loads(tf.extractfile(m).read())
                cells.extend(obj["cells"])
                if len(cells) >= limit:
                    cells = cells[:limit]
                    return cells
    return cells


def bench(fn, iters: int = 3) -> float:
    best = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=64)
    args = ap.parse_args()

    sys.path.insert(0, str(REPO / "src"))
    from odin_tokenizer_fast import OdinFastTokenizer

    fast = OdinFastTokenizer(TOKENIZER_JSON)

    from transformers import PreTrainedTokenizerFast

    hf = PreTrainedTokenizerFast.from_pretrained(str(TOKENIZER_JSON.parent))  # nosec: B615  # local dir, not the Hub
    hf.encode("warmup", add_special_tokens=False)

    cells = load_cells(args.split, args.limit)
    n = len(cells)
    ntok = sum(len(hf.encode(s, add_special_tokens=False)) for s in cells[: min(n, 5000)]) / min(n, 5000)

    # warmup the C word cache
    fast.encode_many(cells[: min(n, 20000)])

    results: dict[str, float] = {}

    def hf_pass():
        for s in cells:
            hf.encode(s, add_special_tokens=False)

    def c_enc_pass():
        for s in cells:
            fast.encode(s)

    def c_many1_pass():
        fast.set_num_threads(1)
        fast.encode_many(cells)

    def c_manyN_pass():
        fast.set_num_threads(args.threads)
        fast.encode_many(cells)

    def c_padded_pass():
        fast.set_num_threads(args.threads)
        fast.encode_padded(cells, args.max_len)

    results["HF encode (baseline)"] = bench(hf_pass)
    results["C encode (per-string)"] = bench(c_enc_pass)
    results["C encode_many (1 thread)"] = bench(c_many1_pass)
    results[f"C encode_many ({args.threads} threads)"] = bench(c_manyN_pass)
    results[f"C encode_padded ({args.threads} threads)"] = bench(c_padded_pass)

    base = results["HF encode (baseline)"]
    print(
        f"\ncorpus: {n} cells from {args.split!r}, ~{ntok:.1f} tokens/cell, "
        f"{sum(len(s) for s in cells) / n:.1f} chars/cell, threads={args.threads}\n"
    )
    print(f"{'variant':<32} {'s':>8} {'cells/s':>10} {'speedup':>9}")
    print("-" * 62)
    for name, dt in results.items():
        print(f"{name:<32} {dt:8.2f} {n / dt:10.0f} {base / dt:8.1f}x")


if __name__ == "__main__":
    main()
