"""Pydantic configuration objects for Odin MLM training."""

import typing
from typing import Any, Literal

import pydantic


class ConfigForDataLoader(pydantic.BaseModel):
    """Configuration for PyTorch ``DataLoader``."""

    model_config = pydantic.ConfigDict(extra="ignore")

    batch_size: int = pydantic.Field(default=8)
    drop_last_batch: bool = pydantic.Field(default=False)
    num_workers: int = pydantic.Field(default=4)
    persistent_workers: bool = pydantic.Field(default=False)
    pin_memory: bool = pydantic.Field(default=True)


class ConfigForMLMCollator(pydantic.BaseModel):
    """Tokenization / MLM augmentation settings."""

    model_config = pydantic.ConfigDict(extra="ignore")

    max_seq_length: int = pydantic.Field(default=512)
    mlm_probability: float = pydantic.Field(default=0.15, ge=0.0, le=1.0)


class ConfigForTextDatamodule(pydantic.BaseModel):
    """Plain-text corpus (one UTF-8 line per segment)."""

    model_config = pydantic.ConfigDict(extra="ignore")

    train_text_path: str = pydantic.Field(
        description="Path to training text (.txt/.gz). One example per non-empty line."
    )
    val_text_path: str | None = pydantic.Field(
        default=None,
        description="Optional validation corpus in the same line-based format.",
    )
    test_text_path: str | None = pydantic.Field(
        default=None,
        description="Optional test corpus for ``trainer.test``.",
    )
    dataloader: ConfigForDataLoader = pydantic.Field(default_factory=ConfigForDataLoader)
    collator: ConfigForMLMCollator = pydantic.Field(default_factory=ConfigForMLMCollator)

    @pydantic.field_validator("train_text_path")
    @classmethod
    def _train_exists(cls, v: str) -> str:
        import pathlib

        if not pathlib.Path(v).is_file():
            raise ValueError(f"Training text path not found or not a file: {v!r}")
        return v


class ConfigForModernBertMLM(pydantic.BaseModel):
    """Loads a pretrained ``ModernBertForMaskedLM`` checkpoint (HF Hub or local dir)."""

    model_config = pydantic.ConfigDict(extra="ignore")

    pretrained_model_name_or_path: str = pydantic.Field(
        description="HF model id (e.g. allenai/modernbert-base) or local directory.",
    )
    trust_remote_code: bool = pydantic.Field(default=False)
    revision: str | None = pydantic.Field(
        default=None,
        description="Optional Hub revision.",
    )


class ConfigForTokenizer(pydantic.BaseModel):
    """Tokenizer sourced from pretrained id/path (defaults match the LM unless overridden)."""

    model_config = pydantic.ConfigDict(extra="ignore")

    pretrained_model_name_or_path: str | None = pydantic.Field(
        default=None,
        description="If set, load tokenizer independently (useful when retokenizing or using a vocab-only json).",
    )
    tokenizer_revision: str | None = pydantic.Field(default=None)


class ConfigForModels(pydantic.BaseModel):
    """Bundles tokenizer + LM configuration."""

    model_config = pydantic.ConfigDict(extra="ignore")

    tokenizer: ConfigForTokenizer = pydantic.Field(default_factory=ConfigForTokenizer)
    modern_bert: ConfigForModernBertMLM = pydantic.Field(description="Masked LM backbone.")


class ConfigForHarness(pydantic.BaseModel):
    """Lightning Trainer + optimizer hyper-parameters."""

    model_config = pydantic.ConfigDict(extra="allow")

    lr: float = pydantic.Field(default=5e-5)
    lr_warmup_ratio: float = pydantic.Field(default=0.1)
    optimizer: str = pydantic.Field(default="AdamW")
    optimizer_kwargs: dict[str, Any] = pydantic.Field(
        default_factory=lambda: {"betas": [0.9, 0.999], "eps": 1e-08, "weight_decay": 0.01},
    )

    precision: str = pydantic.Field(default="bf16-mixed")
    strategy: Literal["auto", "ddp"] = pydantic.Field(default="auto")
    devices: int = pydantic.Field(default=1)
    accelerator: Literal["gpu", "cpu", "cuda", "auto"] = pydantic.Field(default="auto")
    accumulate_grad_batches: int = pydantic.Field(default=1)
    val_check_interval: int | float = pydantic.Field(default=1.0)
    check_val_every_n_epoch: int = pydantic.Field(default=1)
    default_root_dir: str = pydantic.Field(default=".")
    enable_checkpointing: bool = pydantic.Field(default=True)
    gradient_clip_val: float | None = pydantic.Field(default=None)
    limit_train_batches: float = pydantic.Field(default=1.0)
    limit_val_batches: float = pydantic.Field(default=1.0)
    log_every_n_steps: int = pydantic.Field(default=50)
    logger: str = pydantic.Field(default="csv")
    max_epochs: int = pydantic.Field(default=3)
    max_steps: int = pydantic.Field(default=-1)
    max_time: str | None = pydantic.Field(default=None)
    num_nodes: int = pydantic.Field(default=1)
    num_sanity_val_steps: int = pydantic.Field(default=0)
    overfit_batches: float = pydantic.Field(default=0.0)
    reload_dataloaders_every_n_epochs: int = pydantic.Field(default=0)

    checkpoint_path: str | None = pydantic.Field(default=None)


class ConfigForRoot(pydantic.BaseModel):
    """Top-level training configuration file."""

    model_config = pydantic.ConfigDict(extra="forbid")

    datamodule: ConfigForTextDatamodule = pydantic.Field()
    models: ConfigForModels = pydantic.Field()
    training: ConfigForHarness = pydantic.Field()

    @classmethod
    def from_mapping(cls, data: typing.Mapping[str, Any]) -> "ConfigForRoot":
        """Validate a parsed YAML dictionary."""
        return cls.model_validate(data)
