"""Pydantic configuration objects for Odin VAE training."""

from __future__ import annotations

import typing
from typing import Any, Literal

import pydantic

from .augment import SCRIPTS


class ConfigForDataLoader(pydantic.BaseModel):
    """Configuration for the PyTorch ``DataLoader``.

    ``persistent_workers`` must stay ``False``: the resume offset is read
    from the dataset's copy-on-write snapshot at fork time, so workers must
    be rebuilt for the progress to take effect (see ``data/grandwds.py``).
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    batch_size: int = pydantic.Field(default=256, gt=0)
    drop_last_batch: bool = pydantic.Field(default=False)
    num_workers: int = pydantic.Field(default=4, ge=0)
    persistent_workers: bool = pydantic.Field(default=False)
    pin_memory: bool = pydantic.Field(default=True)
    in_order: bool = pydantic.Field(default=True)

    @pydantic.model_validator(mode="after")
    def _check_persistent(self) -> "ConfigForDataLoader":
        if self.persistent_workers:
            raise ValueError("persistent_workers must be False (required for shard-level resume; see grandwds.py).")
        return self


class ConfigForWebDataset(pydantic.BaseModel):
    """Local webdataset shard pattern for one split."""

    model_config = pydantic.ConfigDict(extra="forbid")

    pattern: str = pydantic.Field(
        description="Brace pattern of local shard files, e.g. 'train/shard-{000000..002079}.tar.gz'."
    )
    n_instances_per_shard: int = pydantic.Field(default=2048, gt=0)


class ConfigForDatasetSplit(pydantic.BaseModel):
    """Dataset + dataloader for one split."""

    model_config = pydantic.ConfigDict(extra="forbid")

    dataset: ConfigForWebDataset = pydantic.Field()
    dataloader: ConfigForDataLoader = pydantic.Field(default_factory=ConfigForDataLoader)


class ConfigForAugmentation(pydantic.BaseModel):
    """Per-step cell sampling and letter corruption settings.

    Each training step encodes ``k`` cells (``k`` drawn uniformly from
    ``1..k_input``) sampled uniformly from the cluster — real patent
    families have one surface per member, from one upwards — and decodes
    ``k_latin_target`` latin cells plus ``k_non_latin_target`` non-latin
    cells sampled the same way (stratified to avoid representation bias).
    Each target is tag-primed with probability 1/2 (known-alphabet regime)
    and unprimed otherwise (unknown-alphabet regime). Inputs and targets are
    kept disjoint when the cluster is large enough.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    rate: float = pydantic.Field(default=0.015, ge=0.0, le=1.0, description="Per-letter corruption probability.")
    confusion_weight: float = pydantic.Field(
        default=0.7, ge=0.0, le=1.0, description="P(substitution drawn from the visual-confusion class)."
    )
    k_input: int = pydantic.Field(
        default=4, ge=1, description="Max input cells per cluster; k is drawn uniformly from 1..k_input."
    )
    k_latin_target: int = pydantic.Field(default=4, ge=0)
    k_non_latin_target: int = pydantic.Field(default=2, ge=0)
    max_surface_tokens: int = pydantic.Field(default=128, ge=2)


class ConfigForModel(pydantic.BaseModel):
    """Architecture settings. Encoder, latent and decoder share one dimension."""

    model_config = pydantic.ConfigDict(extra="forbid")

    tokenizer_path: str = pydantic.Field(description="Directory of the trained byte-BPE tokenizer.")
    hidden_size: int = pydantic.Field(default=128, gt=0)
    attention_heads: int = pydantic.Field(default=4, gt=0)
    intermediate_size: int = pydantic.Field(default=512, gt=0)
    encoder_layers: int = pydantic.Field(default=4, ge=1)
    decoder_layers: int = pydantic.Field(default=6, ge=1)
    local_attention: int = pydantic.Field(default=32, ge=2)
    max_position_embeddings: int = pydantic.Field(default=64, ge=2)
    decoder: Literal["modernbert", "t5"] = pydantic.Field(default="modernbert")
    kl_weight: float = pydantic.Field(default=1e-3, ge=0.0)

    @pydantic.model_validator(mode="after")
    def _check_heads(self) -> "ConfigForModel":
        if self.hidden_size % self.attention_heads != 0:
            raise ValueError("hidden_size must be divisible by attention_heads.")
        return self


class ConfigForHarness(pydantic.BaseModel):
    """PyTorch Lightning Trainer + optimizer hyper-parameters."""

    model_config = pydantic.ConfigDict(extra="allow")

    lr: float = pydantic.Field(default=5e-4, gt=0.0)
    lr_warmup_ratio: float = pydantic.Field(default=0.05, ge=0.0, le=1.0)
    optimizer: str = pydantic.Field(default="AdamW")
    optimizer_kwargs: dict[str, Any] = pydantic.Field(
        default_factory=lambda: {"betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01}
    )
    precision: str = pydantic.Field(default="bf16-mixed")
    strategy: Literal["auto", "ddp"] = pydantic.Field(default="auto")
    devices: int = pydantic.Field(default=2, ge=1)
    accelerator: Literal["cpu", "gpu", "cuda", "auto"] = pydantic.Field(default="gpu")
    accumulate_grad_batches: int = pydantic.Field(default=1, ge=1)
    val_check_interval: float = pydantic.Field(default=1.0)
    # 0.0 disables validation entirely.
    limit_val_batches: float = pydantic.Field(default=1.0, ge=0.0)
    gradient_clip_val: float | None = pydantic.Field(default=1.0)
    max_epochs: int = pydantic.Field(default=3, ge=1)
    # Absolute global-step cap (in addition to max_epochs); None disables.
    max_steps: int | None = pydantic.Field(default=None, ge=1)
    enable_checkpointing: bool = pydantic.Field(default=True)
    # Checkpoint every N train steps (in addition to the epoch-end
    # checkpoint). Must be a multiple of n_instances_per_shard / batch_size
    # so the checkpoint lands on a shard boundary (resumable). 0 disables
    # step-based checkpoints (epoch-end only, one file per epoch).
    checkpoint_every_n_steps: int = pydantic.Field(default=0, ge=0)
    # Epoch length in train batches; <= 0 means "auto" (one full pass over the
    # train shards, computed by the entry point from the build manifest).
    limit_train_batches: float = pydantic.Field(default=-1.0)
    log_every_n_steps: int = pydantic.Field(default=50, ge=1)
    generation_log_every_n_steps: int = pydantic.Field(default=500, ge=1)
    n_generation_clusters: int = pydantic.Field(default=3, ge=1)
    checkpoint_path: str | None = pydantic.Field(default=None)


class ConfigForDataModule(pydantic.BaseModel):
    """Root of the shard directory plus the per-split configurations."""

    model_config = pydantic.ConfigDict(extra="forbid")

    data_root: str = pydantic.Field(description="Directory containing the split shard dirs and char_frequencies.json.")
    splits: dict[str, ConfigForDatasetSplit] = pydantic.Field()

    @pydantic.model_validator(mode="after")
    def _check_train(self) -> "ConfigForDataModule":
        if "train" not in self.splits:
            raise ValueError("A 'train' split is required.")
        return self


class ConfigForRoot(pydantic.BaseModel):
    """Top-level training configuration file."""

    model_config = pydantic.ConfigDict(extra="forbid")

    seed: int = pydantic.Field(default=123)
    data_root: str = pydantic.Field(description="Root of the webdataset build (shard dirs + char_frequencies.json).")
    splits: dict[str, ConfigForDatasetSplit] = pydantic.Field()
    augmentation: ConfigForAugmentation = pydantic.Field(default_factory=ConfigForAugmentation)
    model: ConfigForModel = pydantic.Field()
    training: ConfigForHarness = pydantic.Field(default_factory=ConfigForHarness)

    @pydantic.model_validator(mode="after")
    def _check_train(self) -> "ConfigForRoot":
        if "train" not in self.splits:
            raise ValueError("A 'train' split is required.")
        return self

    def datamodule_config(self) -> ConfigForDataModule:
        return ConfigForDataModule(data_root=self.data_root, splits=self.splits)

    @classmethod
    def from_mapping(cls, data: typing.Mapping[str, Any]) -> "ConfigForRoot":
        """Validate a parsed YAML dictionary."""
        return cls.model_validate(data)


def all_scripts() -> tuple[str, ...]:
    return SCRIPTS


__all__ = [
    "ConfigForAugmentation",
    "ConfigForDataLoader",
    "ConfigForDataModule",
    "ConfigForDatasetSplit",
    "ConfigForHarness",
    "ConfigForModel",
    "ConfigForRoot",
    "ConfigForWebDataset",
    "all_scripts",
]
