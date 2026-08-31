"""Fast frozen tokenizer for the Odin byte-level BPE vocabulary.

Drop-in (duck-typed) replacement for the HuggingFace ``PreTrainedTokenizerFast``
loaded from ``/mnt/nvme1/odin_tokenizer``: identical encode/decode semantics,
implemented as a single frozen C extension (``c/odin_tokenizer_fast.c`` +
generated ``c/odin_tokenizer_tables.h``) with a process-wide word cache and a
thread pool for batched encoding.

Build: ``make tokenizer-c`` from the repo root (compiles to
``src/odin_tokenizer_fast/fast.<ext>.so``).

The vocabulary is frozen into the C tables: at init the wrapper verifies the
vocab checksum of ``tokenizer.json`` against the baked-in constant, so a
stale/mismatched tokenizer fails loudly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

from . import fast as _c

DEFAULT_TOKENIZER_JSON = "/mnt/nvme1/odin_tokenizer/tokenizer.json"


def _vocab_checksum(tokenizer_json: Path) -> str:
    pj = json.loads(tokenizer_json.read_text())
    vocab = pj["model"]["vocab"]
    return hashlib.md5("".join(f"{k}{v}" for k, v in sorted(vocab.items())).encode()).hexdigest()  # nosec: B324


class OdinFastTokenizer:
    """Fast tokenizer with the subset of the HF API used by this repo."""

    def __init__(self, tokenizer_json: str | Path = DEFAULT_TOKENIZER_JSON):
        path = Path(tokenizer_json)
        pj = json.loads(path.read_text())
        checksum = _vocab_checksum(path)
        if checksum != _c.VOCAB_CHECKSUM:
            raise ValueError(
                f"Tokenizer vocabulary checksum mismatch (expected {_c.VOCAB_CHECKSUM}, got {checksum}). "
                "Rebuild the C tables: make tokenizer-c-tables."
            )
        self._c = _c
        self._tokenizer_json = str(path)
        added = [t for t in pj.get("added_tokens", []) if t.get("special")]
        self._special_content: dict[str, int] = {t["content"]: t["id"] for t in added}
        self._special_id: dict[int, str] = {t["id"]: t["content"] for t in added}
        self.vocab_size = len(pj["model"]["vocab"])
        self.pad_token_id = self._special_content["[PAD]"]
        self.unk_token_id = self._special_content["[UNK]"]
        self.bos_token_id = self._special_content["[BOS]"]
        self.eos_token_id = self._special_content["[EOS]"]
        self.sep_token_id = self._special_content["[SEP]"]

    # ------------------------------------------------------------------ #
    # encode
    # ------------------------------------------------------------------ #
    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        max_length: int | None = None,
        **kwargs,
    ) -> list[int]:
        """Tokenize one string to a list of ids (HF-identical semantics)."""
        if add_special_tokens:
            raise NotImplementedError(
                "This tokenizer's post-processor adds no special tokens; add_special_tokens=True is always a no-op."
            )
        if kwargs:
            raise TypeError(f"unexpected keyword arguments: {sorted(kwargs)}")
        ids = self._c.encode(text)
        if max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
        return ids

    def encode_many(self, texts: Sequence[str]) -> list[list[int]]:
        """Tokenize many strings; parallelized across CPUs for large inputs."""
        return self._c.encode_many(list(texts))

    def encode_padded(self, texts: Sequence[str], max_len: int) -> tuple:
        """Encode + right-pad in one call (parallel across CPUs).

        Returns ``(ids, raw_lengths)``: ``ids`` is a ``(N, max_len)`` uint16
        numpy array right-padded with ``pad_token_id`` (rows longer than
        ``max_len`` are truncated), ``raw_lengths`` a ``(N,)`` uint16 array of
        the UNTRUNCATED token counts.
        """
        return self._c.encode_padded(list(texts), max_len, self.pad_token_id)

    # ------------------------------------------------------------------ #
    # decode
    # ------------------------------------------------------------------ #
    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return self._c.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def decode_many(self, id_lists: Sequence[Sequence[int]], skip_special_tokens: bool = True) -> list[str]:
        return self._c.decode_many([list(x) for x in id_lists], skip_special_tokens=skip_special_tokens)

    # ------------------------------------------------------------------ #
    # token <-> id
    # ------------------------------------------------------------------ #
    def convert_tokens_to_ids(self, token: str) -> int:
        tid = self._c.token_to_id(token)
        return self.unk_token_id if tid is None else tid

    def convert_ids_to_tokens(self, ids: Sequence[int]) -> str:
        return self._c.decode(list(ids), skip_special_tokens=False)

    # ------------------------------------------------------------------ #
    # thread pool
    # ------------------------------------------------------------------ #
    def set_num_threads(self, n: int) -> None:
        self._c.set_num_threads(n)

    @property
    def num_threads(self) -> int:
        return self._c.num_threads()

    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:  # pragma: no cover
        return f"OdinFastTokenizer(vocab_size={self.vocab_size}, json={self._tokenizer_json})"
