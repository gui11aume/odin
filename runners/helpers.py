"""Shared helper utilities for Lightning runners."""

from __future__ import annotations

import argparse
import datetime
import logging
import re
from pathlib import Path

import lightning.pytorch as pl
import transformers
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import CSVLogger
from transformers import AutoModelForMaskedLM, AutoTokenizer

from odin.config_classes import ConfigForModels, ConfigForTextDatamodule
from odin.data.mlm_datamodule import OdinMLMDataModule

_LOGGING_CONFIGURED = False


def bootstrap_logging(level: int = logging.INFO) -> None:
    """Configure stderr logging once."""
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    _LOGGING_CONFIGURED = True


def experiment_slug(identifier: str) -> str:
    """Sanitize a model id for filesystem paths."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", identifier).strip("_")[:96]


def setup_tokenizer(models_config: ConfigForModels) -> transformers.PreTrainedTokenizerBase:
    """Load tokenizer weights matching ModernBERT vocab."""
    target = (
        models_config.tokenizer.pretrained_model_name_or_path or models_config.modern_bert.pretrained_model_name_or_path
    )
    revision = models_config.tokenizer.tokenizer_revision or models_config.modern_bert.revision
    kwargs = {
        "pretrained_model_name_or_path": target,
        "trust_remote_code": models_config.modern_bert.trust_remote_code,
    }
    if revision:
        kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(**kwargs)  # nosec: B615
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def setup_model(
    models_config: ConfigForModels,
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> transformers.PreTrainedModel:
    """Instantiate ``ModernBertForMaskedLM`` and align embeddings with tokenizer."""
    model = AutoModelForMaskedLM.from_pretrained(  # nosec: B615
        pretrained_model_name_or_path=models_config.modern_bert.pretrained_model_name_or_path,
        trust_remote_code=models_config.modern_bert.trust_remote_code,
        revision=models_config.modern_bert.revision,
    )
    if getattr(model.config, "vocab_size", None) != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    return model


def setup_datamodule(
    dm_config: ConfigForTextDatamodule,
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> OdinMLMDataModule:
    """Create the Lightning ``DataModule``."""
    return OdinMLMDataModule(cfg=dm_config, tokenizer=tokenizer, seed=123)


def setup_callbacks(log_dir: Path) -> list[pl.Callback]:
    """Minimal callback stack mirroring Arena defaults."""
    return [
        RichProgressBar(theme=RichProgressBarTheme(metrics_format=".6g")),
        ModelCheckpoint(
            dirpath=str(log_dir),
            filename="{epoch:03d}",
            auto_insert_metric_name=False,
            save_top_k=-1,
            every_n_epochs=1,
            save_on_train_epoch_end=False,
        ),
    ]


def setup_logger(model_slug: str) -> CSVLogger:
    """Filesystem logger under ``lightning_logs``."""
    stamp = datetime.datetime.now().strftime("%Y-%m-%d")
    return CSVLogger("lightning_logs", name=f"{stamp}_{experiment_slug(model_slug)}")


def setup_profiler(args: argparse.Namespace, log_dir: Path) -> pl.profilers.SimpleProfiler | None:
    """Lightweight profiler hook (CUDA-free) for iterative debugging."""
    if not getattr(args, "do_profile", False):
        return None
    log_dir.mkdir(parents=True, exist_ok=True)
    return pl.profilers.SimpleProfiler(dirpath=str(log_dir))


def persist_hf_checkpoint(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizerBase,
    destination: Path,
) -> None:
    """Serialize HF-compatible artefacts."""
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination)
    tokenizer.save_pretrained(destination)
