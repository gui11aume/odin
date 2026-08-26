"""Lightning datamodule for Hugging Face MLM collation."""

from __future__ import annotations

import pathlib
from typing import Any

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase

from odin.config_classes import ConfigForDataLoader, ConfigForMLMCollator, ConfigForTextDatamodule
from odin.data.text_line_dataset import TextLineDataset


class _MLMBatchCollator:
    """Tokenizes raw strings then applies transformer MLM masking."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, collator_cfg: ConfigForMLMCollator):
        self.tokenizer = tokenizer
        self.collator_cfg = collator_cfg
        self._mlm = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=collator_cfg.mlm_probability,
        )

    def __call__(self, batch_strings: list[str]) -> dict[str, Any]:
        encoded = self.tokenizer(
            batch_strings,
            padding=True,
            truncation=True,
            max_length=self.collator_cfg.max_seq_length,
        )
        features = [{key: encoded[key][i] for key in encoded.keys()} for i in range(len(batch_strings))]
        return self._mlm(features)


class OdinMLMDataModule(pl.LightningDataModule):
    """Wires tokenizer config + deterministic workers for reproducible dataloaders."""

    def __init__(
        self,
        cfg: ConfigForTextDatamodule,
        tokenizer: PreTrainedTokenizerBase,
        seed: int = 123,
    ):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.seed = seed
        self.collator_fn = _MLMBatchCollator(tokenizer=self.tokenizer, collator_cfg=cfg.collator)
        self._train_ds: TextLineDataset | None = None
        self._val_ds: TextLineDataset | None = None
        self._test_ds: TextLineDataset | None = None

    @property
    def has_train(self) -> bool:
        return True  # enforced by pydantic validators

    @property
    def has_test(self) -> bool:
        return self.cfg.test_text_path is not None

    def setup(self, stage: str | None = None) -> None:
        """Materialize splits lazily."""
        if self._train_ds is None:
            self._train_ds = TextLineDataset(self.cfg.train_text_path)
        if stage in {"fit", "validate", None} and self.cfg.val_text_path and self._val_ds is None:
            path = pathlib.Path(self.cfg.val_text_path)
            if not path.is_file():
                raise FileNotFoundError(f"Validation corpus missing: {path}")
            self._val_ds = TextLineDataset(path)
        if stage in {"test", None} and self.cfg.test_text_path and self._test_ds is None:
            path = pathlib.Path(self.cfg.test_text_path)
            if not path.is_file():
                raise FileNotFoundError(f"Test corpus missing: {path}")
            self._test_ds = TextLineDataset(path)

    def train_dataloader(self) -> DataLoader:
        assert self._train_ds is not None
        return self._build_loader(split_cfg=self.cfg.dataloader, dataset=self._train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader | None:
        if not self.cfg.val_text_path:
            return None
        assert self._val_ds is not None
        return self._build_loader(split_cfg=self.cfg.dataloader, dataset=self._val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader | None:
        if not self.cfg.test_text_path:
            return None
        assert self._test_ds is not None
        dl_cfg = self.cfg.dataloader.model_copy(deep=True)
        return self._build_loader(split_cfg=dl_cfg, dataset=self._test_ds, shuffle=False)

    def _build_loader(
        self,
        split_cfg: ConfigForDataLoader,
        dataset: TextLineDataset,
        *,
        shuffle: bool,
    ) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return DataLoader(
            dataset,
            batch_size=split_cfg.batch_size,
            shuffle=shuffle,
            drop_last=split_cfg.drop_last_batch,
            num_workers=split_cfg.num_workers,
            pin_memory=split_cfg.pin_memory,
            persistent_workers=split_cfg.persistent_workers if split_cfg.num_workers > 0 else False,
            collate_fn=self.collator_fn,
            generator=generator,
        )
