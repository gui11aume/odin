"""Data loader with checkpoint-resident shard progress.

The loader is "stateful" in Lightning's sense (it defines ``state_dict`` and
``load_state_dict``), so the shard progress is stored inside the training
checkpoint and restored by Lightning itself on ``fit(ckpt_path=...)``:

- ``state_dict()`` (called in the main process when a checkpoint is saved):
  resolves the current ``(processed_epochs, processed_samples)`` through
  ``progress_fn`` (provided by the data module, backed by the Trainer),
  pushes the resulting whole-shard offset to the dataset, and returns the
  checkpoint state ``{"seed", "processed_epochs", "processed_samples"}``.
- ``load_state_dict()`` (called by Lightning during ``setup_data()`` on
  resume, before the first ``__iter__`` forks workers): validates the seed
  and the shard boundary, then pushes the offset to the dataset.

Legacy checkpoints that predate this format (a seed-only state) leave a
one-shot bootstrap flag: the next ``__iter__`` derives the progress from the
Trainer (the restored loop counters) exactly as the old pipeline did, and
the flag self-extinguishes at the next checkpoint save.

Workers receive the progress through the copy-on-write snapshot of the
dataset taken at fork; the pipeline therefore requires
``multiprocessing_context="fork"`` and ``persistent_workers=False``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch


class ProgressDataset(Protocol):
    """Duck-typed dataset contract (provided by ``GrandWebDataset``)."""

    seed: int | None

    def set_progress(self, processed_epochs: int, processed_shards: int) -> None: ...

    def progress(self) -> tuple[int, int]: ...


class DataLoaderWithAutoCheckpoint(torch.utils.data.DataLoader):
    """DataLoader whose shard progress lives in the Lightning checkpoint.

    Assumes that the dataset has:

    - a ``set_progress(processed_epochs, processed_shards)`` method,
    - a ``progress() -> (processed_epochs, processed_shards)`` method,
    - a ``seed`` attribute (as provided by ``GrandWebDataset``).

    Args (beyond ``DataLoader``):
        progress_fn: Callable returning ``(processed_epochs,
            processed_samples)`` or ``None`` (e.g. outside a fit). Called
            when a checkpoint is saved, and once at the first
            ``__iter__()`` when resuming from a legacy seed-only
            checkpoint.
        n_instances_per_shard: Uniform instances per shard; required for the
            samples<->shards conversion and the shard-boundary validation.
    """

    def __init__(
        self,
        *args,
        progress_fn: Callable[[], tuple[int, int] | None] | None = None,
        n_instances_per_shard: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        for attr in ("set_progress", "progress"):
            if not callable(getattr(self.dataset, attr, None)):
                raise TypeError(f"DataLoaderWithAutoCheckpoint requires a dataset with a {attr}() method.")
        if not hasattr(self.dataset, "seed"):
            raise TypeError("DataLoaderWithAutoCheckpoint requires a dataset with a `seed` attribute.")
        self._dataset: ProgressDataset = self.dataset  # type: ignore[assignment]
        self._progress_fn = progress_fn
        self._n_instances_per_shard = n_instances_per_shard
        self._bootstrap_progress = False

    def __iter__(self):
        """Create a new iterator, bootstrapping progress once for legacy checkpoints."""
        if self._bootstrap_progress:
            self._bootstrap_progress = False
            self._apply_progress(self._resolve_progress())
        return super().__iter__()

    def _resolve_progress(self) -> tuple[int, int] | None:
        if self._progress_fn is None or self._n_instances_per_shard is None:
            return None
        return self._progress_fn()

    def _apply_progress(self, progress: tuple[int, int] | None) -> None:
        """Validate a (epochs, samples) state and push the shard offset to the dataset."""
        if progress is None or self._n_instances_per_shard is None:
            return
        epochs, samples = progress
        n = self._n_instances_per_shard
        if samples % n != 0:
            raise ValueError(f"Cannot resume from a non-shard-boundary state ({samples} samples, shard size {n}).")
        self._dataset.set_progress(epochs, samples // n)

    def state_dict(self) -> dict:
        """Return the checkpoint state (called by Lightning when saving)."""
        self._apply_progress(self._resolve_progress())
        self._bootstrap_progress = False
        state = {"seed": self._dataset.seed}
        if self._n_instances_per_shard is not None:
            epochs, shards = self._dataset.progress()
            state["processed_epochs"] = epochs
            state["processed_samples"] = shards * self._n_instances_per_shard
        return state

    def load_state_dict(self, checkpoint: dict) -> None:
        """Restore state from a checkpoint (called by Lightning on resume)."""
        if "seed" in checkpoint and checkpoint["seed"] != self._dataset.seed:
            raise ValueError(
                f"Checkpoint seed ({checkpoint['seed']}) does not match the configured seed "
                f"({self._dataset.seed}); refusing to resume."
            )
        if "processed_epochs" in checkpoint and "processed_samples" in checkpoint:
            if self._n_instances_per_shard is None:
                raise ValueError("Checkpoint carries shard progress, but this dataloader has no n_instances_per_shard.")
            self._apply_progress((int(checkpoint["processed_epochs"]), int(checkpoint["processed_samples"])))
        else:
            # Legacy seed-only checkpoint: bootstrap from the Trainer next iteration.
            self._bootstrap_progress = True
