"""Tests for the Odin VAE harness helpers."""

from __future__ import annotations

import pytest

from odin_vae.harness import one_cycle_total_steps


@pytest.mark.parametrize(
    ("per_epoch", "epochs", "accum", "expected"),
    [
        (17515, 1, 2, 8758),  # the stage-1 failure: truncation gave 8757, Lightning steps 8758
        (17516, 1, 2, 8758),  # exact division unchanged
        (8757, 3, 1, 26271),  # production 2-GPU shape
        (8757.2, 1, 2, 4379),  # fractional per-epoch batch count
        (1, 1, 4, 1),  # fewer batches than one accumulation group: still one step
        (10, 2, 3, 8),  # ceil(10/3)=4 per epoch, doubled
    ],
)
def test_one_cycle_total_steps(per_epoch: float, epochs: int, accum: int, expected: int) -> None:
    assert one_cycle_total_steps(per_epoch, epochs, accum) == expected


def test_one_cycle_total_steps_matches_scheduler() -> None:
    """The returned total must equal the number of times OneCycleLR can be stepped."""
    import torch
    from torch.optim import AdamW

    per_epoch, epochs, accum = 17515, 1, 2
    total = one_cycle_total_steps(per_epoch, epochs, accum)
    optimizer = AdamW([torch.zeros(1, requires_grad=True)], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=total, pct_start=0.05)
    for _ in range(total):
        scheduler.step()
    with pytest.raises(ValueError, match="total steps"):
        scheduler.step()
