"""Tests for the Odin VAE evaluation runner (pure parts)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

RUNNER_DIR = Path(__file__).resolve().parents[1] / "runners"
if str(RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(RUNNER_DIR))

from evaluate_odin_vae import in_range, key_index, run_val_loss  # noqa: E402


class _FakeCollator:
    """Returns canned tensors, peeling per-batch ce/kl the fake model reads back."""

    def __init__(self, ce_kl_pairs: list[tuple[float, float]], n_per_batch: int):
        self._pairs = ce_kl_pairs
        self._n = n_per_batch
        self.n_truncated = 0

    def __call__(self, batch: list[dict]) -> dict:
        ce, kl = self._pairs.pop(0)
        n_tok = 10
        return {
            "surf_ids": torch.zeros(1, 1, dtype=torch.long),
            "surf_mask": torch.ones(1, 1, dtype=torch.long),
            "k_per_cluster": torch.full((self._n,), 2, dtype=torch.long),
            "target_ids": torch.zeros(self._n, n_tok, dtype=torch.long),
            "target_mask": torch.ones(self._n, n_tok, dtype=torch.long),
            "target_tags": torch.zeros(self._n, dtype=torch.long),
            "target_primed": torch.ones(self._n, dtype=torch.long),
            "_ce": ce,
            "_kl": kl,
        }


class _FakeModel:
    def __call__(self, **tensors) -> dict:
        return {"ce": torch.tensor(tensors.pop("_ce")), "kl": torch.tensor(tensors.pop("_kl"))}


def test_key_index() -> None:
    assert key_index("val-000000000") == 0
    assert key_index("val-000003998.json") == 3998
    assert key_index("train-000000042") == 42


def test_in_range_boundaries() -> None:
    assert in_range("val-000000000", 0, 3998)
    assert in_range("val-000003997", 0, 3998)
    assert not in_range("val-000003998", 0, 3998)
    assert in_range("val-000003998", 3998, 5996)
    assert not in_range("val-000005996", 3998, 5996)


def test_run_val_loss_token_and_cluster_weighting() -> None:
    """CE aggregates token-weighted, KL cluster-weighted across batches."""
    collator = _FakeCollator([(2.0, 0.5), (4.0, 1.0)], n_per_batch=2)
    records = [{"key": f"val-{i:09d}", "tags": ["la"], "cells": ["x"]} for i in range(4)]
    out = run_val_loss(_FakeModel(), collator, records, batch_size=2, kl_weight=1e-3, device="cpu")
    assert out["n_clusters"] == 4
    assert out["n_target_tokens"] == 40
    # mean CE = (2*20 + 4*20) / 40 (each batch: 2 clusters x 10 target tokens)
    assert out["val_ce"] == pytest.approx(3.0)
    # mean KL = (0.5*2 + 1.0*2) / 4
    assert out["val_kl"] == pytest.approx(0.75)
    assert out["val_loss"] == pytest.approx(3.0 + 1e-3 * 0.75)
