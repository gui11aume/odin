"""Tests for the Odin VAE collator."""

from __future__ import annotations

import torch

from odin_vae.augment import LATIN_SCRIPT, SCRIPTS, LetterAugmenter
from odin_vae.data.collators import OdinVAECollator


class FakeTokenizer:
    """One character = one token, over a growing alphabet; tags are known."""

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
        self.eos_token_id = 7
        self._tag_ids = {f"[{tag}]": 1000 + i for i, tag in enumerate(SCRIPTS)}
        self._char_ids: dict[str, int] = {}
        self._char_by_id: dict[int, str] = {}
        self._next = 10

    def _id(self, ch: str) -> int:
        if ch not in self._char_ids:
            self._char_ids[ch] = self._next
            self._char_by_id[self._next] = ch
            self._next += 1
        return self._char_ids[ch]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self._id(ch) for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._char_by_id.get(i, "") for i in ids)

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._tag_ids[token]


def tag_id(tag: str) -> int:
    return 1000 + SCRIPTS.index(tag)


def make_cluster(key: str, cells: list[tuple[str, str]]) -> dict:
    return {"key": key, "tags": [t for t, _ in cells], "cells": [c for _, c in cells]}


def rich_cluster(key: str) -> dict:
    """12 cells: two of each of six scripts."""
    cells = [
        ("la", "a"),
        ("la", "b"),
        ("cn", "c"),
        ("cn", "d"),
        ("gg", "e"),
        ("gg", "f"),
        ("cy", "g"),
        ("cy", "h"),
        ("ab", "i"),
        ("ab", "j"),
        ("jp", "k"),
        ("jp", "l"),
    ]
    return make_cluster(key, cells)


def big_cluster(key: str, n_latin: int = 30, n_non_latin: int = 30) -> dict:
    """Pools large enough that stratified sampling is always disjoint."""
    non_latin_tags = ["cn", "gg", "cy"]
    cells = [(LATIN_SCRIPT, f"a{i:02d}") for i in range(n_latin)]
    cells += [(non_latin_tags[i % 3], f"b{i:02d}") for i in range(n_non_latin)]
    return make_cluster(key, cells)


def make_collator(rate: float = 0.0, **kwargs) -> OdinVAECollator:
    tokenizer = FakeTokenizer()
    augmenter = LetterAugmenter(rate=rate, letter_frequencies={"la": [("x", 1.0)]})
    return OdinVAECollator(tokenizer, augmenter, **kwargs)


def test_batch_shapes() -> None:
    collator = make_collator(k_input=4, k_latin_target=4, k_non_latin_target=2)
    batch = collator([rich_cluster("c0"), rich_cluster("c1"), rich_cluster("c2")])
    assert batch["k_per_cluster"].shape == (3,)
    # k is drawn per cluster from 1..k_input; rows are laid out cluster-major.
    assert batch["k_per_cluster"].min() >= 1
    assert batch["k_per_cluster"].max() <= 4
    assert batch["surf_ids"].shape[0] == int(batch["k_per_cluster"].sum())
    assert batch["surf_mask"].shape == batch["surf_ids"].shape
    assert batch["target_ids"].shape[0] == 3 * 6
    assert batch["target_mask"].shape == batch["target_ids"].shape
    assert batch["target_tags"].shape == (18,)
    assert batch["target_primed"].shape == (18,)
    assert set(batch["target_primed"].tolist()) <= {0, 1}
    # The tag token is prepended to every input row (tag ids start at 1000).
    assert (batch["surf_ids"][:, 0] >= 1000).all()
    assert (batch["surf_mask"][:, 0] == 1).all()


def test_targets_end_with_eos() -> None:
    """Every target row is the surface tokens plus a trailing, masked EOS."""
    collator = make_collator(k_input=4, k_latin_target=2, k_non_latin_target=1)
    batch = collator([rich_cluster("c0")])
    eos = 7
    for row in range(batch["target_ids"].shape[0]):
        n_real = int(batch["target_mask"][row].sum())
        assert n_real >= 2  # at least one surface token + EOS
        assert int(batch["target_ids"][row, n_real - 1]) == eos
        assert int(batch["target_mask"][row, n_real - 1]) == 1


def test_input_count_drawn_uniformly() -> None:
    collator = make_collator(k_input=4)
    ks = []
    for i in range(200):
        batch = collator([rich_cluster(f"c{i}")])
        ks.append(int(batch["k_per_cluster"][0]))
    assert set(ks) == {1, 2, 3, 4}
    # Uniform over 4 values: 200 draws, each value expected ~50.
    from collections import Counter

    counts = Counter(ks)
    assert all(25 <= c <= 75 for c in counts.values())


def test_target_priming_is_coin_flip() -> None:
    collator = make_collator(k_input=4, k_latin_target=4, k_non_latin_target=2)
    primed = []
    for i in range(30):
        batch = collator([rich_cluster(f"c{i}")])
        primed.extend(batch["target_primed"].tolist())
    frac = sum(primed) / len(primed)
    assert len(primed) == 30 * 6
    # Bernoulli(0.5) over 180 draws: comfortably inside 0.35..0.65.
    assert 0.35 < frac < 0.65


def test_stratified_targets() -> None:
    collator = make_collator(k_input=4, k_latin_target=4, k_non_latin_target=2)
    batch = collator([big_cluster("c0")])
    tags = batch["target_tags"].tolist()
    la = sum(1 for t in tags if t == tag_id("la"))
    assert la == 4
    assert len(tags) - la == 2
    # Targets are ordered: latin first, then non-latin.
    assert tags[:4] == [tag_id("la")] * 4


def test_inputs_targets_disjoint_when_possible() -> None:
    collator = make_collator(k_input=4, k_latin_target=4, k_non_latin_target=2)
    batch = collator([big_cluster("c0")])
    tok = collator.tokenizer
    surf_texts = {tok.decode(row.tolist()[1:]) for row in batch["surf_ids"]}
    tgt_texts = {tok.decode(row.tolist()) for row in batch["target_ids"]}
    k = int(batch["k_per_cluster"][0])
    assert len(surf_texts) == k
    assert len(tgt_texts) == 6
    assert surf_texts.isdisjoint(tgt_texts)


def test_backfill_when_latin_scarce() -> None:
    # One latin cell: the 4 latin target slots are backfilled from non-latin.
    cells = [
        ("la", "a"),
        ("cn", "b"),
        ("cn", "c"),
        ("cn", "d"),
        ("cn", "e"),
        ("gg", "f"),
        ("gg", "g"),
        ("gg", "h"),
        ("gg", "i"),
        ("gg", "j"),
        ("gg", "k"),
        ("gg", "l"),
    ]
    cluster = make_cluster("c0", cells)
    collator = make_collator(k_input=4, k_latin_target=4, k_non_latin_target=2)
    batch = collator([cluster])
    tags = batch["target_tags"].tolist()
    assert sum(1 for t in tags if t == tag_id("la")) <= 1
    assert len(tags) == 6


def test_cjk_cells_never_corrupted() -> None:
    collator = make_collator(rate=1.0, k_input=2, k_latin_target=0, k_non_latin_target=2)
    cells = [("la", "x"), ("cn", "\u5f20\u4e09"), ("cn", "\u738b\u4e94"), ("la", "y")]
    cluster = make_cluster("c0", cells)
    batch = collator([cluster])
    cn_cells = {"\u5f20\u4e09", "\u738b\u4e94"}
    tok = collator.tokenizer
    for row, tag in zip(batch["target_ids"], batch["target_tags"].tolist()):
        text = tok.decode(row.tolist())
        if tag == tag_id("cn"):
            assert text in cn_cells  # byte-identical, no corruption
        else:  # backfilled latin cell, corrupted to the table letter
            assert text == "x"


def test_corruption_applied_to_inputs_and_targets() -> None:
    collator = make_collator(rate=1.0, k_input=1, k_latin_target=1, k_non_latin_target=0)
    cluster = make_cluster("c0", [("la", "abcdefgh")])
    batch = collator([cluster])
    tok = collator.tokenizer
    input_text = tok.decode(batch["surf_ids"][0].tolist()[1:])
    target_text = tok.decode(batch["target_ids"][0].tolist())
    # Both the input and the (same, backfilled) cell were corrupted: with
    # rate=1.0 no original letter survives (table holds 'x' only).
    assert input_text != "abcdefgh"
    assert target_text != "abcdefgh"


def test_deterministic_within_same_batch_position() -> None:
    collator = make_collator(rate=0.5)
    cluster = rich_cluster("c0")
    batch1 = collator([cluster])
    collator._batch_idx = 0  # reset the position counter for reproducibility
    batch2 = collator([cluster])
    for key in ("surf_ids", "surf_mask", "k_per_cluster", "target_ids", "target_mask", "target_tags", "target_primed"):
        assert torch.equal(batch1[key], batch2[key])


def test_max_tokens_truncation() -> None:
    collator = make_collator(k_input=1, k_latin_target=1, k_non_latin_target=0, max_tokens=4)
    cluster = make_cluster("c0", [("la", "abcdefgh")])
    batch = collator([cluster])
    assert batch["surf_ids"].shape[1] <= 4  # tag + at most 3 tokens
    assert collator.n_truncated >= 1


def test_rng_seed_includes_key_and_position() -> None:
    collator = make_collator()
    same_a1 = [collator._rng_for(0, 0, "key-A").random() for _ in range(5)]
    same_a2 = [collator._rng_for(0, 0, "key-A").random() for _ in range(5)]
    other_b = [collator._rng_for(0, 0, "key-B").random() for _ in range(5)]
    other_pos = [collator._rng_for(0, 1, "key-A").random() for _ in range(5)]
    assert same_a1 == same_a2
    assert same_a1 != other_b
    assert same_a1 != other_pos


def test_small_cluster_fallback_to_overlap() -> None:
    # 2 cells total: inputs take both, targets must overlap (backfill).
    collator = make_collator(k_input=2, k_latin_target=2, k_non_latin_target=0)
    cluster = make_cluster("c0", [("la", "a"), ("la", "b")])
    batch = collator([cluster])
    k = int(batch["k_per_cluster"][0])
    assert k in (1, 2)
    assert batch["surf_ids"].shape[0] == k
    assert batch["target_ids"].shape[0] == 2  # exactly k targets, even with overlap
    tok = collator.tokenizer
    assert {tok.decode(r.tolist()[1:]) for r in batch["surf_ids"]} <= {"a", "b"}
    assert all(tok.decode(r.tolist()) in {"a", "b"} for r in batch["target_ids"])


def test_all_latin_cluster_still_yields_full_targets() -> None:
    # No non-latin cells at all: the non-latin slots are backfilled from latin.
    cells = [("la", chr(ord("a") + i)) for i in range(12)]
    cluster = make_cluster("c0", cells)
    collator = make_collator(k_input=4, k_latin_target=4, k_non_latin_target=2)
    batch = collator([cluster])
    assert batch["target_ids"].shape[0] == 6
    assert (batch["target_tags"] == tag_id("la")).all()


def test_single_cell_cluster() -> None:
    cluster = make_cluster("c0", [("la", "a")])
    collator = make_collator(k_input=4, k_latin_target=4, k_non_latin_target=2)
    batch = collator([cluster])
    assert batch["surf_ids"].shape[0] == 1
    assert batch["target_ids"].shape[0] == 6  # all the same cell, by overlap
