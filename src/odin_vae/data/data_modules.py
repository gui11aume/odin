"""Lightning DataModule over local webdataset shards (no intermediate store).

Each shard sample already contains the full cluster record, so the stream
goes directly to the collator. Shard-level resume is checkpoint-resident:
the train dataloader is stateful (see ``DataLoaderWithAutoCheckpoint``), so
the ``(epoch, shard)`` offset is written into the Lightning checkpoint when
it is saved and restored by Lightning on ``fit(ckpt_path=...)``. The
contract requires a uniform ``n_instances_per_shard`` per split and
checkpoints on shard boundaries; the seed is a config constant, validated
against the checkpoint on resume.
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

        The train split is checkpoint-resident: its progress is resolved
        from the Trainer when a checkpoint is saved and restored by
        Lightning on resume. Val/test are finite (``is_endless=False``) and
        always start at shard 0 (their fixed ``seed`` shuffle is consumed in
        full on every pass, so no progress tracking is needed).
        """
        _mp_context = torch.multiprocessing.get_context("fork") if dataloader_config.num_workers > 0 else None
        is_train = split == "train"
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
            progress_fn=(lambda: self._data_progress()) if is_train else None,
            n_instances_per_shard=self.config.splits[split].dataset.n_instances_per_shard if is_train else None,
        )

    def _data_progress(self) -> tuple[int, int] | None:
        """Current ``(processed_epochs, processed_samples)``, or None outside a fit."""
        if self.trainer is None:
            return None
        epoch_progress = self.trainer.fit_loop.epoch_progress.current
        batch_progress = self.trainer.fit_loop.epoch_loop.batch_progress

        # getattr: Lightning's internal progress types are version-fragile for static analysis.
        processed_epochs = int(getattr(epoch_progress, "processed", 0))
        processed_batches = int(getattr(batch_progress.current, "processed", 0))

        # When a checkpoint is saved on the last train batch, Lightning can
        # serialize a boundary state where ``batch_progress.is_last_batch``
        # is true while ``epoch_progress.processed`` still points to the
        # previous epoch. Normalize this boundary to the next-epoch start so
        # resume begins at shard 0 of the following epoch.
        if bool(batch_progress.is_last_batch):
            processed_epochs = max(processed_epochs, int(getattr(epoch_progress, "ready", 0)))
            processed_batches = 0

        batch_size = self.config.splits["train"].dataloader.batch_size
        return (processed_epochs, processed_batches * batch_size)

    def train_dataloader(self) -> DataLoaderWithAutoCheckpoint | None:
        """Return the training dataloader."""
        return self._ensure_loader("train") if "train" in self.config.splits else None

    def val_dataloader(self) -> DataLoaderWithAutoCheckpoint | None:
        """Return the validation dataloader."""
        return self._ensure_loader("val") if "val" in self.config.splits else None

    def test_dataloader(self) -> DataLoaderWithAutoCheckpoint | None:
        """Return the test dataloader."""
        return self._ensure_loader("test") if "test" in self.config.splits else None
