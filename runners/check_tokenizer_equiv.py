"""Verify a fast tokenizer implementation against the HuggingFace tokenizer.

Streams the WebDataset shards (train/val/test), extracts every cell value, and
compares the candidate encoder against ``PreTrainedTokenizerFast.encode``
(default ``add_special_tokens=False``), plus a decode roundtrip.

Reference algorithm (mirrors tokenizers 0.22.2 Rust pipeline exactly):
  1. added-vocabulary extraction: literal leftmost-longest scan for the 17
     special tokens ([PAD] ... [am]); each match emits its id directly.
  2. GPT-2 ByteLevel regex chunking on each remaining segment
     (apostrophe alts | ?L+ | ?N+ | ?other+ | \\s+(?!\\S) | \\s+, with the
     \\s+(?!\\S) backtracking that splits interior whitespace runs).
  3. Byte-level BPE per chunk: symbols = single-byte token ids, then the
     iterative merge-list algorithm (min-heap on (rank, pos), stale-entry
     validation), as in tokenizers' ``Word::merge_all``.

Usage:
    python runners/check_tokenizer_equiv.py [--split train|val|test|all]
                                             [--encoder hf|ref|fast]
                                             [--limit N] [--seed N]
"""

from __future__ import annotations

import argparse
import heapq
import json
import random
import sys
import tarfile
import time
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WDS_ROOT = Path("/mnt/nvme1/odin_wds_v3")
TOKENIZER_JSON = Path("/mnt/nvme1/odin_tokenizer/tokenizer.json")


# ---------------------------------------------------------------------------
# Reference algorithm (pure Python, mirrors the Rust pipeline)
# ---------------------------------------------------------------------------
class RefEncoder:
    def __init__(self, tokenizer_json: Path):
        pj = json.loads(tokenizer_json.read_text())
        self.vocab = pj["model"]["vocab"]
        merges = pj["model"]["merges"]

        bs = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
        b2c = {b: chr(b) for b in bs}
        n = 0
        for b in range(256):
            if b not in b2c:
                b2c[b] = chr(256 + n)
                n += 1
        c2b = {c: b for b, c in b2c.items()}
        self.byte_id = {c2b[k]: v for k, v in self.vocab.items() if len(k) == 1}
        assert len(self.byte_id) == 256

        self.merge: dict[tuple[int, int], tuple[int, int]] = {}
        for rank, (a, b) in enumerate(merges):
            self.merge[(self.vocab[a], self.vocab[b])] = (rank, self.vocab[a + b])

        self.specials = [(k, v) for k, v in self.vocab.items() if v <= 16]
        assert len(self.specials) == 17

        # Unicode property tables for the regex
        self._cat = {}

    def _cls(self, c: str) -> int:
        cat = self._cat.get(c)
        if cat is None:
            ucat = unicodedata.category(c)
            if ucat in ("Lu", "Ll", "Lt", "Lm", "Lo"):
                cat = 1
            elif ucat in ("Nd", "Nl", "No"):
                cat = 2
            else:
                cp = ord(c)
                if (
                    c in " \t\n\r\x0b\x0c"
                    or cp in (0x85, 0xA0, 0x1680, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000)
                    or (0x2000 <= cp <= 0x200A)
                ):
                    cat = 3
                else:
                    cat = 0
            self._cat[c] = cat
        return cat

    def encode(self, s: str) -> list[int]:
        out: list[int] = []
        # 1. special-token extraction (all specials: 5 ASCII chars, start '[')
        i, n = 0, len(s)
        specials = self.specials
        while i < n:
            if s[i] == "[":
                hit = None
                for k, v in specials:
                    if s.startswith(k, i):
                        hit = (v, len(k))
                        break
                if hit is not None:
                    out.append(hit[0])
                    i += hit[1]
                    continue
            j = i
            while j < n:
                if s[j] == "[" and any(s.startswith(k, j) for k, _ in specials):
                    break
                j += 1
            for ch in regex_chunks_ref(s, i, j, self._cls):
                out.extend(self._bpe_chunk(ch))
            i = j
        return out

    def _bpe_chunk(self, chunk: str) -> list[int]:
        symbols = [self.byte_id[bb] for bb in chunk.encode("utf-8")]
        n = len(symbols)
        nxt = list(range(1, n)) + [-1]
        alive = [True] * n
        heap: list[tuple[int, int, int]] = []
        merge = self.merge

        def push(item):
            heapq.heappush(heap, item)

        for i in range(n - 1):
            m = merge.get((symbols[i], symbols[i + 1]))
            if m:
                push((m[0], i, m[1]))
        while heap:
            _rank, pos, new_id = heapq.heappop(heap)
            if not alive[pos]:
                continue
            pnext = nxt[pos]
            if pnext == -1 or not alive[pnext]:
                continue
            m = merge.get((symbols[pos], symbols[pnext]))
            if m is None or m[1] != new_id:
                continue
            symbols[pos] = new_id
            alive[pnext] = False
            nxt[pos] = nxt[pnext]
            prev_i = pos - 1
            while prev_i >= 0 and not alive[prev_i]:
                prev_i -= 1
            if prev_i >= 0:
                m = merge.get((symbols[prev_i], symbols[pos]))
                if m:
                    push((m[0], prev_i, m[1]))
            nnext = nxt[pos]
            if nnext != -1:
                m = merge.get((symbols[pos], symbols[nnext]))
                if m:
                    push((m[0], pos, m[1]))
        return [symbols[i] for i in range(n) if alive[i]]


def regex_chunks_ref(s: str, a: int, b: int, cls) -> list[str]:
    """Chunk s[a:b] with the GPT-2 ByteLevel regex semantics."""
    chunks: list[str] = []
    i = a
    n = b
    while i < n:
        c = s[i]
        if c == "'":
            m = None
            for alt in ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d"):
                if s.startswith(alt, i):
                    m = len(alt)
                    break
            if m:
                chunks.append(s[i : i + m])
                i += m
                continue
            j = i
            while j < n and cls(s[j]) not in (1, 2, 3):
                j += 1
            chunks.append(s[i:j])
            i = j
            continue
        if cls(c) == 3:
            # alternatives in regex order: " ?L+" | " ?N+" | " ?other+"
            # (a space followed by a run of that class is ONE chunk)
            if i + 1 < n and cls(s[i + 1]) in (1, 2, 0):
                t2 = cls(s[i + 1])
                jj = i + 1
                while jj < n and cls(s[jj]) == t2:
                    jj += 1
                chunks.append(s[i:jj])
                i = jj
                continue
            # else: next is whitespace or end of string -> \s+ alternatives.
            # \s+(?!\S) backtracks so the match stops one space short of a
            # non-space char: interior run of k spaces -> chunk of k-1;
            # trailing run -> whole run.
            j = i
            while j < n and cls(s[j]) == 3:
                j += 1
            k = j - i
            if j < n:
                chunks.append(s[i : i + k - 1])
                i += k - 1
            else:
                chunks.append(s[i:j])
                i = j
            continue
        t = cls(c)
        j = i + 1
        while j < n and cls(s[j]) == t:
            j += 1
        chunks.append(s[i:j])
        i = j
    return chunks


# ---------------------------------------------------------------------------
# Corpus streaming
# ---------------------------------------------------------------------------
def iter_cells(split: str, limit: int | None = None):
    root = WDS_ROOT / split
    count = 0
    for shard in sorted(root.glob("*.tar.gz")):
        with tarfile.open(shard) as tf:
            for m in tf.getmembers():
                if not m.isfile() or not m.name.endswith(".json"):
                    continue
                obj = json.loads(tf.extractfile(m).read())
                for cell in obj["cells"]:
                    yield cell
                    count += 1
                    if limit is not None and count >= limit:
                        return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--encoder", default="fast", choices=["ref", "fast"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample", type=int, default=None, help="random sample of N cells (with --seed)")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    ref = RefEncoder(TOKENIZER_JSON)

    from transformers import PreTrainedTokenizerFast

    hf = PreTrainedTokenizerFast.from_pretrained(str(TOKENIZER_JSON.parent))  # nosec: B615  # local dir, not the Hub
    hf.encode("warmup", add_special_tokens=False)

    if args.encoder == "ref":
        cand = ref.encode
        name = "ref(python)"
    else:
        sys.path.insert(0, str(REPO / "src"))
        from odin_tokenizer_fast import OdinFastTokenizer  # noqa: E402

        fast = OdinFastTokenizer(TOKENIZER_JSON)
        cand = fast.encode
        name = "fast(c)"

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    total = mism = roundtrip_fail = 0
    t0 = time.perf_counter()
    for split in splits:
        cells = list(iter_cells(split, args.limit))
        if args.sample is not None:
            rng = random.Random(args.seed)
            cells = rng.sample(cells, min(args.sample, len(cells)))
        for s in cells:
            mine = cand(s)
            want = hf.encode(s, add_special_tokens=False)
            if list(mine) != want:
                mism += 1
                if mism <= 5:
                    print(f"MISMATCH [{split}]: {s!r}")
                    print(f"  {name}: {list(mine)}")
                    print(f"  hf    : {want}")
            if hf.decode(want, skip_special_tokens=True) != s:
                roundtrip_fail += 1
        total += len(cells)
        print(f"[{split}] {len(cells)} cells, running mismatches: {mism}", flush=True)
    dt = time.perf_counter() - t0
    print(
        f"done: {total} cells in {dt:.1f}s ({total / dt:.0f} cells/s), "
        f"mismatches={mism}, decode-roundtrip-failures={roundtrip_fail}"
    )
    sys.exit(1 if mism else 0)


if __name__ == "__main__":
    main()
