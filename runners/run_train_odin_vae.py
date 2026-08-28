"""Lightning entry point for Odin VAE training.

Usage:
    CUDA_VISIBLE_DEVICES=1,2 python runners/run_train_odin_vae.py --config runners/odin_vae_config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import lightning.pytorch as pl
import torch
import yaml
from lightning.pytorch import strategies
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import CSVLogger
from transformers import PreTrainedTokenizerFast

RUNNER_DIR = Path(__file__).resolve().parent
if str(RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(RUNNER_DIR))

from helpers import bootstrap_logging  # noqa: E402
from odin.harness import Trainer  # noqa: E402
from odin_vae.augment import SCRIPTS, LetterAugmenter  # noqa: E402
from odin_vae.config_classes import ConfigForRoot  # noqa: E402
from odin_vae.data.collators import OdinVAECollator  # noqa: E402
from odin_vae.data.data_modules import OdinVAEDataModule  # noqa: E402
from odin_vae.harness import OdinVAELightningHarness  # noqa: E402
from odin_vae.model import OdinModel  # noqa: E402

log = logging.getLogger(__name__)


def _map_accelerator(name: str) -> str:
    return "cuda" if name.lower() == "gpu" else name


def _resolve_strategy(harness_cfg) -> strategies.DDPStrategy | str:
    if harness_cfg.strategy.lower() == "ddp":
        return strategies.DDPStrategy()
    return "auto"


def _setup_tokenizer(path: str) -> PreTrainedTokenizerFast:
    tokenizer = PreTrainedTokenizerFast.from_pretrained(path)  # nosec: B615  # local directory, not the Hub
    if tokenizer.pad_token_id is None or tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError(f"Tokenizer at {path} must define pad/bos/eos special tokens.")
    for tag in SCRIPTS:
        token = f"[{tag}]"
        if tokenizer.convert_tokens_to_ids(token) == tokenizer.unk_token_id:
            raise ValueError(f"Tag token {token} missing from the tokenizer vocabulary.")
    return tokenizer


def _per_rank_batches(manifest: dict, batch_size: int, devices: int) -> float:
    """Train batches per rank for one full pass over the train shards."""
    n_shards = manifest["n_train_shards"]
    n_train = manifest["n_train"]
    shard_size = manifest["shard_size"]
    best = 0
    for rank in range(devices):
        instances = sum(min(shard_size, n_train - shard * shard_size) for shard in range(rank, n_shards, devices))
        best = max(best, (instances + batch_size - 1) // batch_size)
    return float(best)


def _load_letter_frequencies(data_root: str) -> dict:
    path = Path(data_root) / "char_frequencies.json"
    if not path.is_file():
        raise FileNotFoundError(f"Letter-frequency table not found: {path} (run the shard builder first).")
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    bootstrap_logging(logging.INFO)

    parser = argparse.ArgumentParser(description="Train the Odin name-surface VAE.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        raw_yaml = yaml.safe_load(handle)
    root_cfg = ConfigForRoot.from_mapping(raw_yaml)

    pl.seed_everything(root_cfg.seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    tokenizer = _setup_tokenizer(root_cfg.model.tokenizer_path)
    model = OdinModel(
        root_cfg.model,
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    log.info("Model: %d parameters (decoder=%s)", sum(p.numel() for p in model.parameters()), root_cfg.model.decoder)

    aug_cfg = root_cfg.augmentation
    augmenter = LetterAugmenter(aug_cfg.rate, _load_letter_frequencies(root_cfg.data_root), aug_cfg.confusion_weight)
    collator = OdinVAECollator(
        tokenizer,
        augmenter,
        k_input=aug_cfg.k_input,
        k_latin_target=aug_cfg.k_latin_target,
        k_non_latin_target=aug_cfg.k_non_latin_target,
        max_tokens=aug_cfg.max_surface_tokens,
        seed=root_cfg.seed,
    )
    datamodule = OdinVAEDataModule(root_cfg.datamodule_config(), collator=collator, seed=root_cfg.seed)

    train_cfg = root_cfg.splits["train"]
    limit_train_batches: float = root_cfg.training.limit_train_batches
    if limit_train_batches > 0 and float(limit_train_batches).is_integer():
        limit_train_batches = int(limit_train_batches)  # int => absolute batch count, not a fraction
    if limit_train_batches <= 0:
        manifest_path = Path(root_cfg.data_root) / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Build manifest not found: {manifest_path} (run the shard builder first).")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        limit_train_batches = _per_rank_batches(manifest, train_cfg.dataloader.batch_size, root_cfg.training.devices)
        log.info(
            "limit_train_batches (auto): %s per rank for %d devices", limit_train_batches, root_cfg.training.devices
        )

    harness = OdinVAELightningHarness(
        model=model,
        tokenizer=tokenizer,
        val_shard_pattern=f"{root_cfg.data_root}/{root_cfg.splits['val'].dataset.pattern}",
        n_generation_clusters=root_cfg.training.n_generation_clusters,
        generation_log_every_n_steps=root_cfg.training.generation_log_every_n_steps,
    )

    training_cfg = root_cfg.training
    ckpt_path = Path(training_cfg.checkpoint_path) if training_cfg.checkpoint_path else None
    if training_cfg.checkpoint_path and (ckpt_path is None or not ckpt_path.is_file()):
        raise FileNotFoundError(f"checkpoint_path not found: {training_cfg.checkpoint_path}")

    csv_logger = CSVLogger("lightning_logs", name="odin_vae")
    callbacks: list[pl.Callback] = [RichProgressBar(theme=RichProgressBarTheme(metrics_format=".6g"))]
    if training_cfg.enable_checkpointing:
        checkpoint_kwargs = dict(dirpath=str(csv_logger.log_dir), auto_insert_metric_name=False, save_top_k=-1)
        # Epoch-end checkpoint (saved on the last train batch of each epoch).
        callbacks.append(
            ModelCheckpoint(
                filename="{epoch:03d}",
                every_n_epochs=1,
                save_on_train_epoch_end=False,
                **checkpoint_kwargs,
            )
        )
        # Optional mid-epoch checkpoints; must land on shard boundaries so the
        # checkpoint is resumable.
        if training_cfg.checkpoint_every_n_steps > 0:
            n_instances = root_cfg.splits["train"].dataset.n_instances_per_shard
            batch_size = root_cfg.splits["train"].dataloader.batch_size
            if n_instances % batch_size != 0:
                raise ValueError("n_instances_per_shard must be a multiple of batch_size for checkpointing.")
            batches_per_shard = n_instances // batch_size
            if training_cfg.checkpoint_every_n_steps % batches_per_shard != 0:
                raise ValueError(
                    f"checkpoint_every_n_steps ({training_cfg.checkpoint_every_n_steps}) must be a multiple of the "
                    f"batches per shard ({batches_per_shard} = n_instances_per_shard / batch_size) so checkpoints "
                    "land on shard boundaries."
                )
            callbacks.append(
                ModelCheckpoint(
                    filename="{epoch:03d}-step{step:06d}",
                    every_n_train_steps=training_cfg.checkpoint_every_n_steps,
                    **checkpoint_kwargs,
                )
            )
    trainer = Trainer(
        lr=training_cfg.lr,
        lr_warmup_ratio=training_cfg.lr_warmup_ratio,
        optimizer=training_cfg.optimizer,
        optimizer_kwargs=training_cfg.optimizer_kwargs,
        accelerator=_map_accelerator(training_cfg.accelerator),
        strategy=_resolve_strategy(training_cfg),
        precision=training_cfg.precision,
        devices=training_cfg.devices,
        gradient_clip_val=training_cfg.gradient_clip_val,
        accumulate_grad_batches=training_cfg.accumulate_grad_batches,
        val_check_interval=training_cfg.val_check_interval,
        limit_train_batches=limit_train_batches,
        limit_val_batches=training_cfg.limit_val_batches,
        max_epochs=training_cfg.max_epochs,
        # Lightning rejects max_steps=None; -1 (its default) means unbounded.
        max_steps=training_cfg.max_steps if training_cfg.max_steps is not None else -1,
        enable_checkpointing=training_cfg.enable_checkpointing,
        log_every_n_steps=training_cfg.log_every_n_steps,
        logger=csv_logger,
        callbacks=callbacks,
    )

    if trainer.is_global_zero:
        snapshot_dir = Path(csv_logger.log_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        (snapshot_dir / "config.snapshot.yaml").write_text(yaml.safe_dump(raw_yaml), encoding="utf-8")

    trainer.fit(model=harness, datamodule=datamodule, ckpt_path=ckpt_path)
    log.info("Training complete.")
