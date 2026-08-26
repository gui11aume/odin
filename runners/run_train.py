"""Lightning entrypoint mirroring Arena runners for ModernBERT MLM."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import lightning.pytorch as pl
import torch
import yaml
from lightning.pytorch import strategies

RUNNER_DIR = Path(__file__).resolve().parent
if str(RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(RUNNER_DIR))

from helpers import (  # noqa: E402
    bootstrap_logging,
    experiment_slug,
    persist_hf_checkpoint,
    setup_callbacks,
    setup_datamodule,
    setup_logger,
    setup_model,
    setup_profiler,
    setup_tokenizer,
)
from odin.config_classes import ConfigForHarness, ConfigForRoot  # noqa: E402
from odin.harness import MLMLightningHarness, Trainer  # noqa: E402


def _map_accelerator(name: str) -> str:
    if name.lower() == "gpu":
        return "cuda"
    return name


def _resolve_strategy(
    harness_cfg: ConfigForHarness,
) -> strategies.DDPStrategy | str:
    if harness_cfg.strategy.lower() == "ddp":
        return strategies.DDPStrategy()
    return "auto"


def _checkpoint_path(cfg_path: str | None) -> str | None:
    if not cfg_path:
        return None
    resolved = Path(cfg_path)
    if not resolved.is_file():
        raise FileNotFoundError(f"checkpoint_path not found: {resolved}")
    return str(resolved)


def _persist_model(
    harness: MLMLightningHarness,
    tokenizer,
    csv_logger,
    *,
    output_dir: Path | None,
) -> None:
    if not csv_logger.log_dir:
        return
    persist_hf_checkpoint(
        model=harness.model,
        tokenizer=tokenizer,
        destination=output_dir or Path(csv_logger.log_dir) / "hf_checkpoint",
    )


if __name__ == "__main__":
    bootstrap_logging(logging.INFO)

    parser = argparse.ArgumentParser(description="Train ModernBERT with masked LM")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--do-profile", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional HF export directory overriding the default lightning log path.",
    )
    args = parser.parse_args()

    pl.seed_everything(123)
    torch.set_float32_matmul_precision("medium")

    with open(args.config, encoding="utf-8") as handle:
        raw_yaml = yaml.safe_load(handle)

    root_cfg = ConfigForRoot.from_mapping(raw_yaml)
    dm_cfg = root_cfg.datamodule
    models_cfg = root_cfg.models
    harness_cfg = root_cfg.training

    tokenizer_obj = setup_tokenizer(models_cfg)
    language_model = setup_model(models_cfg, tokenizer_obj)
    datamodule = setup_datamodule(dm_cfg, tokenizer_obj)

    slug = experiment_slug(models_cfg.modern_bert.pretrained_model_name_or_path)
    csv_logger = setup_logger(slug)
    log_dir = Path(csv_logger.log_dir) if csv_logger.log_dir else Path.cwd()
    callbacks = setup_callbacks(log_dir)
    profiler = setup_profiler(args, log_dir)

    lightning_harness = MLMLightningHarness(model=language_model)

    ckpt_resume = _checkpoint_path(harness_cfg.checkpoint_path)

    lightning_trainer = Trainer(
        lr=harness_cfg.lr,
        lr_warmup_ratio=harness_cfg.lr_warmup_ratio,
        optimizer=harness_cfg.optimizer,
        optimizer_kwargs=harness_cfg.optimizer_kwargs,
        accelerator=_map_accelerator(harness_cfg.accelerator),
        strategy=_resolve_strategy(harness_cfg),
        precision=harness_cfg.precision,
        devices=harness_cfg.devices,
        num_nodes=harness_cfg.num_nodes,
        gradient_clip_val=harness_cfg.gradient_clip_val,
        accumulate_grad_batches=harness_cfg.accumulate_grad_batches,
        num_sanity_val_steps=harness_cfg.num_sanity_val_steps,
        check_val_every_n_epoch=harness_cfg.check_val_every_n_epoch,
        val_check_interval=harness_cfg.val_check_interval,
        default_root_dir=harness_cfg.default_root_dir,
        enable_checkpointing=harness_cfg.enable_checkpointing,
        logger=csv_logger,
        log_every_n_steps=harness_cfg.log_every_n_steps,
        max_epochs=harness_cfg.max_epochs,
        max_steps=harness_cfg.max_steps,
        max_time=harness_cfg.max_time,
        limit_train_batches=harness_cfg.limit_train_batches,
        limit_val_batches=harness_cfg.limit_val_batches,
        overfit_batches=harness_cfg.overfit_batches,
        reload_dataloaders_every_n_epochs=harness_cfg.reload_dataloaders_every_n_epochs,
        callbacks=callbacks,
        profiler=profiler,
    )

    if lightning_trainer.is_global_zero and csv_logger.log_dir is not None:
        snapshot_dir = Path(csv_logger.log_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        (snapshot_dir / "config.snapshot.yaml").write_text(yaml.safe_dump(raw_yaml), encoding="utf-8")

    if datamodule.has_train:
        lightning_trainer.fit(model=lightning_harness, datamodule=datamodule, ckpt_path=ckpt_resume)
        out_dir = Path(args.output_dir) if args.output_dir else None
        if lightning_trainer.is_global_zero:
            _persist_model(lightning_harness, tokenizer_obj, csv_logger, output_dir=out_dir)

    if datamodule.has_test:
        lightning_trainer.test(model=lightning_harness, datamodule=datamodule)
