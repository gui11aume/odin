"""Collator: sample, corrupt, and tokenize cluster records into model batches.

For each cluster in the batch the collator:

1. samples ``k_input`` cells uniformly at random -> encoder inputs,
2. samples ``k_latin_target`` latin cells + ``k_non_latin_target`` non-latin
   cells -> decoder targets (stratified, disjoint from the inputs when the
   cluster is large enough; backfilled otherwise),
3. corrupts the letters of the selected cells (per the LetterAugmenter; CJK
   cells are never corrupted),
4. tokenizes (tag token prepended for inputs only) and right-pads.

The per-sample RNG is seeded from the global seed, the worker id, a
batch counter, and a hash of the cluster key: corruption and sampling are
reproducible and independent of the batch composition.
"""

from __future__ import annotations

import hashlib
import random

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
        k_latin_target: int = 4,
        k_non_latin_target: int = 2,
        max_tokens: int = 128,
        seed: int = 123,
    ):
        self.tokenizer = tokenizer
        self.augmenter = augmenter
        self.k_input = k_input
        self.k_latin_target = k_latin_target
        self.k_non_latin_target = k_non_latin_target
        self.max_tokens = max_tokens
        self.seed = seed
        self.pad_token_id = tokenizer.pad_token_id
        self._tag_ids = {tag: tokenizer.convert_tokens_to_ids(f"[{tag}]") for tag in SCRIPTS}
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

        Preference order: (1) cells of ``pool`` disjoint from ``used``,
        (2) cells of ``fallback_pool`` disjoint from ``used``, (3) any cell
        with overlap allowed. This guarantees exactly ``k`` targets per
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

        The targets are always exactly ``k_latin_target + k_non_latin_target``
        (the model relies on a fixed count per cluster); each target keeps
        its own true script tag.
        """
        input_idx = rng.sample(range(n), min(self.k_input, n))
        latin = [j for j in range(n) if tags[j] == LATIN_SCRIPT]
        non_latin = [j for j in range(n) if tags[j] != LATIN_SCRIPT]
        any_cell = list(range(n))
        used = set(input_idx)
        targets = self._pick(latin, used, self.k_latin_target, non_latin, any_cell, rng)
        used.update(targets)
        targets += self._pick(non_latin, used, self.k_non_latin_target, latin, any_cell, rng)
        return input_idx, targets

    def _tokenize(self, text: str, tag: str | None) -> list[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if tag is not None:
            ids = [self._tag_ids[tag]] + ids
        if len(ids) > self.max_tokens:
            ids = ids[: self.max_tokens]
            self.n_truncated += 1
        return ids

    # ------------------------------------------------------------------ #
    def __call__(self, examples: list[dict]) -> dict:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0

        surface_rows: list[tuple[str, str]] = []
        target_rows: list[tuple[str, str]] = []
        for position, example in enumerate(examples):
            tags: list[str] = example["tags"]
            cells: list[str] = example["cells"]
            n = len(cells)
            rng = self._rng_for(worker_id, position, example["key"])
            input_idx, target_idx = self._select(tags, n, rng)
            for j in input_idx:
                surface_rows.append((tags[j], self.augmenter.corrupt(tags[j], cells[j], rng)))
            for j in target_idx:
                target_rows.append((tags[j], self.augmenter.corrupt(tags[j], cells[j], rng)))
        self._batch_idx += 1

        surf_ids = [self._tokenize(text, tag) for tag, text in surface_rows]
        tgt_ids = [self._tokenize(text, None) for tag, text in target_rows]
        tgt_tags = [self._tag_ids[tag] for tag, _ in target_rows]

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
            "target_ids": tgt_ids_t,
            "target_mask": tgt_mask_t,
            "target_tags": torch.tensor(tgt_tags, dtype=torch.long),
            "n_clusters": len(examples),
        }
