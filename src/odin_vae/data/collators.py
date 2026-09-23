"""Collator: sample, corrupt, and tokenize cluster records into model batches.

For each cluster in the batch the collator:

1. draws `k` uniformly from `1..k_input` and samples `k` cells uniformly
   at random -> encoder inputs (real patent families have one surface per
   member, from 1 upwards),
2. samples `k_latin_target` latin cells + `k_non_latin_target` non-latin
   cells -> decoder targets (stratified, disjoint from the inputs when the
   cluster is large enough; backfilled otherwise),
3. flips a coin per target: with probability 1/2 the target is tag-primed
   (the decoder is seeded with the script tag, as in inference with a known
   alphabet), otherwise it is unprimed (decode from the latent alone, as in
   log-probability computation with an unknown alphabet),
4. corrupts the letters of the selected cells (per the LetterAugmenter; CJK
   cells are never corrupted),
5. tokenizes (tag token prepended for inputs only), appends the EOS token to
   every target (the decoder must learn where a surface ends), and right-pads.

The per-sample RNG is seeded from the global seed, the worker id, a
batch counter, and a hash of the cluster key: corruption and sampling are
reproducible and independent of the batch composition.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np
import torch
from torch.utils.data import get_worker_info

from ..augment import LATIN_SCRIPT, SCRIPTS, LetterAugmenter


class OdinVAECollator:
    """Collate a list of cluster records into the tensors consumed by OdinModel."""

    def __init__(
        self,
        tokenizer,
        augmenter: LetterAugmenter,
        *,
        k_input: int = 4,
        k_input_weights: list[float] | None = None,
        k_latin_target: int = 4,
        k_non_latin_target: int = 2,
        max_tokens: int = 128,
        seed: int = 123,
    ):
        if k_input_weights is not None and len(k_input_weights) != k_input:
            raise ValueError(f"k_input_weights has {len(k_input_weights)} entries but k_input is {k_input}.")
        self.tokenizer = tokenizer
        self.augmenter = augmenter
        self.k_input = k_input
        self.k_input_weights = k_input_weights
        self.k_latin_target = k_latin_target
        self.k_non_latin_target = k_non_latin_target
        self.max_tokens = max_tokens
        self.seed = seed
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self._tag_ids = {tag: tokenizer.convert_tokens_to_ids(f"[{tag}]") for tag in SCRIPTS}
        # Fast tokenizers expose a vectorized encode_padded(texts, max_len)
        # that encodes + right-pads in one (multi-threaded) C call. When it is
        # available we build the token tensors from numpy directly, which is
        # the hot path of training. Otherwise we fall back to per-string
        # encode + Python padding (the classic behavior).
        self._fast = hasattr(tokenizer, "encode_padded")
        self._batch_idx = 0
        self.n_truncated = 0

    # ------------------------------------------------------------------ #
    def _rng_for(self, worker_id: int, position: int, key: str) -> random.Random:
        digest = hashlib.sha1(  # nosec: B324  # non-security use: deterministic augmentation seeding
            f"{self.seed}:{worker_id}:{self._batch_idx}:{position}:{key}".encode("utf-8"), usedforsecurity=False
        ).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))  # nosec: B311  # deterministic, non-security RNG

    @staticmethod
    def _pick(
        pool: list[int],
        used: set[int],
        k: int,
        fallback_pool: list[int],
        any_pool: list[int],
        rng: random.Random,
    ) -> list[int]:
        """Sample exactly k cells.

        Preference order: (1) cells of `pool` disjoint from `used`,
        (2) cells of `fallback_pool` disjoint from `used`, (3) any cell
        with overlap allowed. This guarantees exactly `k` targets per
        cluster even for degenerate clusters (few cells, one script only).
        """
        if k == 0 or not any_pool:
            return []
        available = [j for j in pool if j not in used]
        out = list(rng.sample(available, min(k, len(available))))
        shortfall = k - len(out)
        if shortfall > 0 and fallback_pool:
            fb_available = [j for j in fallback_pool if j not in used]
            take = rng.sample(fb_available, min(shortfall, len(fb_available)))
            out += take
            shortfall -= len(take)
        if shortfall > 0:
            out += rng.choices(any_pool, k=shortfall)  # overlap allowed
        return out

    def _select(self, tags: list[str], n: int, rng: random.Random) -> tuple[list[int], list[int]]:
        """Return (input indices, target indices) for one cluster.

        The number of inputs is drawn from `1..k_input` — uniformly by default,
        or with the configured `k_input_weights` (a random subset of the
        `k_input` candidates, so the disjointness from the targets —
        computed against the full candidate set — is preserved).
        The targets are always exactly `k_latin_target + k_non_latin_target`
        (the model relies on a fixed count per cluster); each target keeps
        its own true script tag.
        """
        input_candidates = rng.sample(range(n), min(self.k_input, n))
        latin = [j for j in range(n) if tags[j] == LATIN_SCRIPT]
        non_latin = [j for j in range(n) if tags[j] != LATIN_SCRIPT]
        any_cell = list(range(n))
        used = set(input_candidates)
        targets = self._pick(latin, used, self.k_latin_target, non_latin, any_cell, rng)
        used.update(targets)
        targets += self._pick(non_latin, used, self.k_non_latin_target, latin, any_cell, rng)
        # weights=None => uniform over 1..k_input (matches the classic behavior)
        k = rng.choices(range(1, self.k_input + 1), weights=self.k_input_weights)[0]
        input_idx = rng.sample(input_candidates, min(k, len(input_candidates)))
        return input_idx, targets

    def _tokenize(self, text: str, tag: str | None) -> list[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if tag is not None:
            ids = [self._tag_ids[tag]] + ids
        if len(ids) > self.max_tokens:
            ids = ids[: self.max_tokens]
            self.n_truncated += 1
        return ids

    def _fast_encode_rows(
        self, tags: list[str], texts: list[str], *, with_tag: bool, with_eos: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Vectorized encode + pad, bit-identical to the per-string path.

        Semantics matched to `_tokenize`:
          * with_tag:  row = [tag_id] + enc(text), truncated to max_tokens
            (a tag counts against the budget, so text gets max_tokens - 1).
          * with_eos:  row = enc(text) truncated to max_tokens, then + eos.
        The returned width is the max row length in the batch (like the
        legacy `pad`), right-padded with pad_token_id, mask 1 on real tokens.
        """
        n = len(texts)
        if with_tag:
            budget = self.max_tokens - 1
        else:
            budget = self.max_tokens
        buf, raw = self.tokenizer.encode_padded(texts, budget)
        raw = raw.astype(np.int64)
        clip = np.minimum(raw, budget)  # text tokens actually kept
        self.n_truncated += int((raw > budget).sum())
        row_len = clip + (1 if with_tag else 0) + (1 if with_eos else 0)
        if n == 0:
            return (
                torch.empty((0, 0), dtype=torch.long),
                torch.empty((0, 0), dtype=torch.long),
            )
        width = int(row_len.max())
        ids_np = np.full((n, width), self.pad_token_id, dtype=np.uint16)
        if with_tag:
            ids_np[:, 0] = [self._tag_ids[t] for t in tags]
            fill = min(width - 1, budget)
            if fill > 0:
                ids_np[:, 1 : 1 + fill] = buf[:, :fill]
        else:
            fill = min(width, budget)
            if fill > 0:
                ids_np[:, :fill] = buf[:, :fill]
            if with_eos:
                ids_np[np.arange(n), clip.astype(np.intp)] = self.eos_token_id
        idx = np.arange(width)[None, :]
        mask_np = (idx < row_len[:, None]).astype(np.int64)
        return torch.from_numpy(ids_np).to(torch.long), torch.from_numpy(mask_np).to(torch.long)

    # ------------------------------------------------------------------ #
    def __call__(self, examples: list[dict]) -> dict:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0

        surface_rows: list[tuple[str, str]] = []
        target_rows: list[tuple[str, str]] = []
        target_primed: list[int] = []
        k_per_cluster: list[int] = []
        for position, example in enumerate(examples):
            tags: list[str] = example["tags"]
            cells: list[str] = example["cells"]
            n = len(cells)
            rng = self._rng_for(worker_id, position, example["key"])
            input_idx, target_idx = self._select(tags, n, rng)
            k_per_cluster.append(len(input_idx))
            for j in input_idx:
                surface_rows.append((tags[j], self.augmenter.corrupt(tags[j], cells[j], rng)))
            for j in target_idx:
                # 50% tag-primed (alphabet known) / 50% unprimed (alphabet unknown).
                target_primed.append(1 if rng.random() < 0.5 else 0)
                target_rows.append((tags[j], self.augmenter.corrupt(tags[j], cells[j], rng)))
        self._batch_idx += 1

        tgt_tags = [self._tag_ids[tag] for tag, _ in target_rows]

        if self._fast:
            surf_ids_t, surf_mask_t = self._fast_encode_rows(
                [tag for tag, _ in surface_rows], [text for _, text in surface_rows], with_tag=True, with_eos=False
            )
            tgt_ids_t, tgt_mask_t = self._fast_encode_rows(
                [tag for tag, _ in target_rows], [text for _, text in target_rows], with_tag=False, with_eos=True
            )
        else:
            surf_ids = [self._tokenize(text, tag) for tag, text in surface_rows]
            tgt_ids = [self._tokenize(text, None) + [self.eos_token_id] for tag, text in target_rows]

            def pad(rows: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
                length = max(len(row) for row in rows)
                ids = torch.full((len(rows), length), self.pad_token_id, dtype=torch.long)
                mask = torch.zeros((len(rows), length), dtype=torch.long)
                for i, row in enumerate(rows):
                    ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
                    mask[i, : len(row)] = 1
                return ids, mask

            surf_ids_t, surf_mask_t = pad(surf_ids)
            tgt_ids_t, tgt_mask_t = pad(tgt_ids)
        return {
            "surf_ids": surf_ids_t,
            "surf_mask": surf_mask_t,
            "k_per_cluster": torch.tensor(k_per_cluster, dtype=torch.long),
            "target_ids": tgt_ids_t,
            "target_mask": tgt_mask_t,
            "target_tags": torch.tensor(tgt_tags, dtype=torch.long),
            "target_primed": torch.tensor(target_primed, dtype=torch.long),
        }
