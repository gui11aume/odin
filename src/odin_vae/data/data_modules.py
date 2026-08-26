"""Lightning DataModule over local webdataset shards (no intermediate store).

Each shard sample already contains the full cluster record, so the stream
goes directly to the collator. The shard-level resume contract is preserved:
uniform ``n_instances_per_shard`` per split, checkpoints on shard boundaries,
and the seed restored from the checkpoint.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
import webdataset as wds

from ..config_classes import ConfigForDataLoader, ConfigForDataModule
from .adapters import ClusterSampleAdapter
from .data_loaders import DataLoaderWithAutoCheckpoint


class OdinVAEDataModule(pl.LightningDataModule):
    """DataModule for the Odin VAE training splits.

    Example config:

    .. code-block:: yaml

       data_root: /mnt/nvme1/odin_wds
       splits:
         train:
           dataset:
             pattern: "train/shard-{000000..002079}.tar.gz"
             n_instances_per_shard: 2048
           dataloader:
             batch_size: 256
             num_workers: 4
         val:
           dataset:
             pattern: "val/shard-{000000..000001}.tar.gz"
             n_instances_per_shard: 2048
           dataloader:
             batch_size: 256
             num_workers: 2
    """

    def __init__(
        self,
        config: dict[str, Any] | ConfigForDataModule,
        collator: Any,
        seed: int | None = None,
    ):
        """Initialize the data module.

        Args:
            config: The data module configuration (dict or ConfigForDataModule).
            collator: The collator for the data loaders.
            seed: The random seed for shard shuffling (None = no shuffling).
        """
        super().__init__()
        self.config = config if isinstance(config, ConfigForDataModule) else ConfigForDataModule.model_validate(config)
        if self.config.splits["train"].dataloader.persistent_workers:
            raise ValueError("persistent_workers must be False (required for shard-level resume).")
        self.collator = collator
        self.seed = seed
        # Caches for datasets and dataloaders by split.
        self.datasets: dict[str, ClusterSampleAdapter] = {}
        self.loaders: dict[str, DataLoaderWithAutoCheckpoint] = {}

    def _split_pattern(self, split: str) -> str:
        pattern = self.config.splits[split].dataset.pattern
        return (
            pattern
            if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:|^/|^\.?\.?/", pattern)
            else f"{self.config.data_root}/{pattern}"
        )

    def prepare_data(self) -> None:
        """Ensure the shard files of every configured split exist."""
        for split in self.config.splits:
            pattern = self._split_pattern(split)
            files = wds.shardlists.expand_urls(pattern) if "{" in pattern else [pattern]
            if not files:
                raise ValueError(f"No shard files match the '{split}' pattern: {pattern!r}.")
            for path in files[:3] + files[-3:]:
                if not Path(path).is_file():
                    raise ValueError(f"Shard file not found for '{split}' split: {path!r}")

    def _ensure_dataset(self, split: str) -> None:
        """Create and cache the dataset for a split if needed."""
        if split in self.datasets:
            return
        if split not in self.config.splits:
            raise ValueError(f"No '{split}' split specified in the config.")
        self.datasets[split] = ClusterSampleAdapter(
            urls=self._split_pattern(split),
            seed=self.seed,
            is_endless=split == "train",
        )

    def _ensure_loader(self, split: str) -> DataLoaderWithAutoCheckpoint:
        """Create and cache the dataloader for the given split."""
        if split in self.loaders:
            return self.loaders[split]
        self._ensure_dataset(split)
        dataset = self.datasets[split]
        dataloader_config = self.config.splits[split].dataloader
        loader = self.create_dataloader(dataset, dataloader_config, split=split)
        self.loaders[split] = loader
        return loader

    def setup(self, stage: str) -> None:
        """Setup the datasets for the stage (train+val for 'fit', else the split)."""
        if stage == "fit":
            self._ensure_dataset("train")
            if "val" in self.config.splits:
                self._ensure_dataset("val")
        elif stage in ("validate", "test", "predict"):
            self._ensure_dataset(stage)

    def teardown(self, stage: str | None = None) -> None:
        """Clear cached datasets and dataloaders."""
        if stage in ("fit", None):
            for key in ("train", "val"):
                self.datasets.pop(key, None)
                self.loaders.pop(key, None)
        elif stage in ("validate", "test", "predict", None):
            self.datasets.pop(stage, None)
            self.loaders.pop(stage, None)

    def create_dataloader(
        self,
        dataset: ClusterSampleAdapter,
        dataloader_config: ConfigForDataLoader,
        *,
        split: str,
    ) -> DataLoaderWithAutoCheckpoint:
        """Create a dataloader from a dataset and a dataloader config.

        Train resumes in-shard via the processed-shard offset. Val/test are
        finite (``is_endless=False``): applying the train shard offset would
        skip past the short eval shard lists and yield no batches (especially
        after a checkpoint resume when train progress is non-zero), so they
        only sync the epoch for the shuffle and always start at shard 0.
        """
        progress_fn = (
            self._trainer_epoch_batch_progress if split == "train" else self._finite_split_progress_from_trainer
        )
        _mp_context = torch.multiprocessing.get_context("fork") if dataloader_config.num_workers > 0 else None
        return DataLoaderWithAutoCheckpoint(
            dataset=dataset,
            batch_size=dataloader_config.batch_size,
            collate_fn=self.collator,
            drop_last=dataloader_config.drop_last_batch,
            num_workers=dataloader_config.num_workers,
            persistent_workers=dataloader_config.persistent_workers,
            pin_memory=dataloader_config.pin_memory,
            multiprocessing_context=_mp_context,
            in_order=dataloader_config.in_order,
            progress_fn=progress_fn,
        )

    def _trainer_epoch_batch_progress(self) -> tuple[int, int]:
        """Read processed epoch and in-epoch processed batches from the Trainer.

        When a checkpoint is saved on the last train batch, Lightning can
        serialize a boundary state where ``batch_progress.is_last_batch`` is
        true while ``epoch_progress.processed`` still points to the previous
        epoch. We normalize this boundary to the next-epoch start so resume
        begins at shard 0 of the following epoch.
        """
        if self.trainer is None:
            return (0, 0)
        epoch_progress = self.trainer.fit_loop.epoch_progress.current
        batch_progress = self.trainer.fit_loop.epoch_loop.batch_progress

        # getattr: Lightning's internal progress types are version-fragile for static analysis.
        processed_epochs = int(getattr(epoch_progress, "processed", 0))
        processed_batches = int(getattr(batch_progress.current, "processed", 0))

        if bool(batch_progress.is_last_batch):
            processed_epochs = max(processed_epochs, int(getattr(epoch_progress, "ready", 0)))
            processed_batches = 0

        n_instances_per_shard = self.config.splits["train"].dataset.n_instances_per_shard
        batch_size = self.config.splits["train"].dataloader.batch_size
        processed_shards = int(batch_size * processed_batches / n_instances_per_shard)
        return (processed_epochs, processed_shards)

    def _finite_split_progress_from_trainer(self) -> tuple[int, int]:
        """Epoch for the shard shuffle; shard offset 0 for val/test streams."""
        processed_epochs, _ = self._trainer_epoch_batch_progress()
        return (processed_epochs, 0)

    def train_dataloader(self) -> DataLoaderWithAutoCheckpoint | None:
        """Return the training dataloader."""
        return self._ensure_loader("train") if "train" in self.config.splits else None

    def val_dataloader(self) -> DataLoaderWithAutoCheckpoint | None:
        """Return the validation dataloader."""
        return self._ensure_loader("val") if "val" in self.config.splits else None

    def test_dataloader(self) -> DataLoaderWithAutoCheckpoint | None:
        """Return the test dataloader."""
        return self._ensure_loader("test") if "test" in self.config.splits else None

    def load_checkpoint(self, checkpoint: dict) -> None:
        """Load a checkpoint (validates the shard boundary and recovers the seed).

        The information about the epoch and the number of processed batches
        is restored in the trainer and is directly sent to the dataset.
        """
        fit_loop = checkpoint["loops"]["fit_loop"]
        # Check that the checkpoint is on a shard boundary.
        epoch_progress = fit_loop["epoch_loop.batch_progress"]
        if not bool(epoch_progress["is_last_batch"]):
            processed_batches = epoch_progress["current"]["processed"]
            batch_size = self.config.splits["train"].dataloader.batch_size
            n_instances_per_shard = self.config.splits["train"].dataset.n_instances_per_shard
            if processed_batches * batch_size % n_instances_per_shard != 0:
                raise ValueError("Checkpoint is not on a shard boundary.")
        # Load the random seed.
        fit_loop_dict = fit_loop["state_dict"]
        train_dataloader_state = fit_loop_dict["combined_loader"][0]
        self.seed = train_dataloader_state["seed"]
