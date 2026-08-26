"""Train the Odin byte-level BPE tokenizer on inventor name surfaces.

Corpus
------
Input is a (phase-3 cleaned) cluster corpus: one cluster per line,
tab-separated ``tag{value}`` cells across the twelve scripts. Only the
cell *values* (the name surfaces) are fed to the trainer — the ``tag{...}``
container syntax is stripped first, so it never pollutes the vocabulary.

Design (fixed for the Odin problem)
-----------------------------------
* Byte-level BPE: every Unicode string encodes/decodes losslessly, including
  spellings absent from the training corpus (OCR noise, rare CJK). WordPiece
  was rejected because characters outside a capped alphabet become [UNK]
  irreversibly — fatal for a decoder that must emit character-exact names.
* No normalizer (identity): the contract is byte-exact surface handling —
  every corpus surface must decode back to itself, and real-world inputs
  (OCR, web data) regularly contain canonically decomposed sequences that
  must not be silently rewritten. NFD was rejected outright (it decomposes
  composed Hangul syllables into jamo); NFC was rejected after verification
  found 529 non-NFC cells (decomposed Greek/Devanagari/Turkish marks) that
  it would rewrite, breaking roundtrip identity.
* ``ByteLevel(add_prefix_space=False)`` + ``decoders.ByteLevel()``: the
  default prefix-space behavior leaves a stray leading space on decode.
* ``BpeTrainer(initial_alphabet=ByteLevel().alphabet())``: without it, bytes
  absent from the training corpus are silently dropped from encodings.
* Case-sensitive: ALL-CAPS variants are part of the data.
* Special tokens for the seq2seq / set-encoder setup:
  [PAD] [UNK] [BOS] [EOS] [SEP] (surface separator for exchangeable
  multi-input encoders) and one tag token per script ([la] ... [am]).

Verification
------------
A second full pass over the corpus asserts encode->decode identity on every
cell, counts [UNK] (must be 0) and reports per-script tokens-per-name.

Usage:
    python src/train_odin_tokenizer.py INPUT [-o OUTPUT_DIR]
                                       [--vocab-size 16384]
                                       [--max-clusters N] [--skip-verify]
"""

from __future__ import annotations

import argparse
import gzip
import logging
import sys
import time
from collections.abc import Iterator
from pathlib import Path

from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

log = logging.getLogger(__name__)

TAGS = ("la", "cy", "gk", "ab", "cn", "jp", "kr", "dv", "hb", "th", "gg", "am")
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]"] + [f"[{t}]" for t in TAGS]
DEFAULT_VOCAB_SIZE = 16384
DEFAULT_OUTPUT_DIR = "odin_tokenizer"


def _valid_cell(cell: str) -> bool:
    return len(cell) >= 4 and cell[2] == "{" and cell.endswith("}") and cell[:2] in TAGS and cell[3:-1] != ""


def iter_cell_values(path: Path, max_clusters: int | None = None) -> Iterator[str]:
    """Yield each cell value of the cluster corpus as one training document.

    Structurally malformed cells (defensive; the input is normally a
    phase-3 cleaned file) are skipped.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    clusters = 0
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if max_clusters is not None and clusters >= max_clusters:
                return
            cells = [cell for cell in line.rstrip("\r\n").split("\t") if _valid_cell(cell)]
            if not cells:
                continue
            clusters += 1
            for cell in cells:
                yield cell[3:-1]


def build_tokenizer() -> Tokenizer:
    """Empty tokenizer skeleton with the fixed Odin pre/decode settings."""
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    # No normalizer on purpose: encode/decode must be byte-exact for any input.
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    return tok


def train(
    toker: Tokenizer,
    path: Path,
    *,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    max_clusters: int | None = None,
) -> int:
    """Train the BPE model on the cell values of ``path``; return cell count."""
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel().alphabet(),
    )
    count = 0

    def _counting() -> Iterator[str]:
        nonlocal count
        for value in iter_cell_values(path, max_clusters):
            count += 1
            yield value

    toker.train_from_iterator(_counting(), trainer=trainer)
    return count


def vocab_report(toker: Tokenizer) -> dict:
    """Static vocab sanity: size, special tokens, container-syntax leakage.

    Single-byte brace pieces ``{``/``}`` are part of the universal byte
    alphabet and are expected; the leakage signal is a *merged* (multi-byte)
    piece containing a brace, which can only come from the ``tag{value}``
    container syntax being fed to the trainer.
    """
    vocab = toker.get_vocab()
    return {
        "vocab_size": len(vocab),
        "missing_specials": [t for t in SPECIAL_TOKENS if t not in vocab],
        "pieces_with_braces": [p for p in vocab if ("{" in p or "}" in p) and len(p) > 1],
    }


def verify(toker: Tokenizer, path: Path, *, chunk_size: int = 65536, max_clusters: int | None = None) -> dict:
    """Full-corpus roundtrip check plus per-script token statistics.

    Returns dict with total_cells, roundtrip_failures, unk_tokens, and
    per-tag [token_count, cell_count] pairs.
    """
    unk_id = toker.token_to_id("[UNK]")
    per_tag: dict[str, list[int]] = {t: [0, 0, 0] for t in TAGS}  # tokens, cells, max tokens
    total_cells = 0
    failures = 0
    unk_tokens = 0

    pending_vals: list[str] = []
    pending_tags: list[str] = []

    def flush() -> None:
        nonlocal failures, unk_tokens
        if not pending_vals:
            return
        encodings = toker.encode_batch(pending_vals)
        decoded = toker.decode_batch([e.ids for e in encodings])
        for val, enc, dec in zip(pending_vals, encodings, decoded):
            if dec != val:
                failures += 1
                if failures <= 3:
                    log.warning("roundtrip mismatch: %r -> %r", val, dec)
            unk_tokens += sum(1 for i in enc.ids if i == unk_id)
        for tag, enc in zip(pending_tags, encodings):
            per_tag[tag][0] += len(enc.ids)
            per_tag[tag][1] += 1
            per_tag[tag][2] = max(per_tag[tag][2], len(enc.ids))
        pending_vals.clear()
        pending_tags.clear()

    opener = gzip.open if path.suffix == ".gz" else open
    clusters = 0
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if max_clusters is not None and clusters >= max_clusters:
                break
            cells = [cell for cell in line.rstrip("\r\n").split("\t") if _valid_cell(cell)]
            if not cells:
                continue
            clusters += 1
            for cell in cells:
                total_cells += 1
                pending_tags.append(cell[:2])
                pending_vals.append(cell[3:-1])
                if len(pending_vals) >= chunk_size:
                    flush()
        flush()

    return {
        "total_cells": total_cells,
        "roundtrip_failures": failures,
        "unk_tokens": unk_tokens,
        "per_tag": per_tag,
    }


def save_tokenizer(toker: Tokenizer, output_dir: Path) -> None:
    """Persist an HF-loadable tokenizer (tokenizer.json + tokenizer_config.json)."""
    hf = PreTrainedTokenizerFast(tokenizer_object=toker)
    hf.pad_token = "[PAD]"
    hf.unk_token = "[UNK]"
    hf.bos_token = "[BOS]"
    hf.eos_token = "[EOS]"
    output_dir.mkdir(parents=True, exist_ok=True)
    hf.save_pretrained(output_dir)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    parser = argparse.ArgumentParser(description="Train the Odin byte-level BPE tokenizer.")
    parser.add_argument("input", help="Cluster corpus (.txt or .txt.gz), normally the phase-3 cleaned file.")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for the saved HF tokenizer (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=None,
        help="Train on only the first N clusters (quick iteration).",
    )
    parser.add_argument("--skip-verify", action="store_true", help="Skip the full-corpus roundtrip pass.")
    parser.add_argument("--verify-chunk", type=int, default=65536, help="Batch size for the verification pass.")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(f"Input not found: {input_path}")
    output_dir = Path(args.output_dir)

    toker = build_tokenizer()

    t0 = time.perf_counter()
    n_cells = train(toker, input_path, vocab_size=args.vocab_size, max_clusters=args.max_clusters)
    log.info(
        "trained on %d cell values in %.1fs (vocab size %d)", n_cells, time.perf_counter() - t0, toker.get_vocab_size()
    )

    report = vocab_report(toker)
    if report["missing_specials"]:
        raise SystemExit(f"Missing special tokens in vocab: {report['missing_specials']}")
    if report["pieces_with_braces"]:
        raise SystemExit(f"Container syntax leaked into the vocabulary: {report['pieces_with_braces'][:10]}")
    log.info("vocab ok: %d pieces, %d special tokens, no tag-syntax leakage", report["vocab_size"], len(SPECIAL_TOKENS))

    if not args.skip_verify:
        t0 = time.perf_counter()
        stats = verify(toker, input_path, chunk_size=args.verify_chunk, max_clusters=args.max_clusters)
        elapsed = time.perf_counter() - t0
        log.info(
            "verified %d cells in %.1fs: roundtrip failures=%d, unk_tokens=%d",
            stats["total_cells"],
            elapsed,
            stats["roundtrip_failures"],
            stats["unk_tokens"],
        )
        for tag in TAGS:
            tokens, cells, max_tokens = stats["per_tag"][tag]
            if cells:
                log.info(
                    "  %-2s mean %.2f tokens/name, max %d tokens, %d cells", tag, tokens / cells, max_tokens, cells
                )
        if stats["roundtrip_failures"] or stats["unk_tokens"]:
            raise SystemExit("Verification failed: tokenizer is not lossless on the corpus.")

    save_tokenizer(toker, output_dir)
    log.info("saved HF tokenizer to %s", output_dir)


if __name__ == "__main__":
    main()
