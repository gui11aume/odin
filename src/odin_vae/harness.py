"""PyTorch Lightning harness for Odin VAE training.

Adds, on top of the plain VAE forward, the arena-style OneCycle optimizer
setup and a rank-0 generation smoke check: a fixed handful of clusters is
encoded and decoded once per 12 script tags after each validation epoch (and
every N train steps), so the log directory accumulates a readable sample of
what the latent currently produces per script.
"""

from __future__ import annotations

import datetime
import json
import math
import tarfile
from pathlib import Path
from typing import Any, cast

import lightning.pytorch as pl
import torch
import webdataset as wds
from lightning.pytorch.utilities.types import LRSchedulerConfig
from torch import optim
from torch.optim.optimizer import Optimizer

from odin.harness import Trainer

from .augment import SCRIPTS
from .model import OdinModel

_TAG_PREFIX = "["


def one_cycle_total_steps(per_epoch_batches: float, max_epochs: int, accumulate_grad_batches: int) -> int:
    """Optimizer steps OneCycleLR must cover.

    Lightning steps the scheduler per optimizer step, and a trailing partial
    accumulation group at the end of the epoch limit still triggers a step, so
    each epoch contributes ``ceil(per_epoch / accum)`` steps. Truncating the
    division undercounts by one whenever the division is inexact (crash:
    ``Tried to step N times. The specified number of total steps is N-1``).
    """
    return max(1, math.ceil(per_epoch_batches / accumulate_grad_batches) * max(1, max_epochs))


class OdinVAELightningHarness(pl.LightningModule):
    """Wraps an ``OdinModel`` and logs loss components + periodic generations."""

    def __init__(
        self,
        model: OdinModel,
        tokenizer,
        *,
        val_shard_pattern: str | None = None,
        n_generation_clusters: int = 3,
        generation_max_new_tokens: int = 48,
        generation_temperature: float = 0.0,
        generation_log_every_n_steps: int = 500,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.val_shard_pattern = val_shard_pattern
        self.n_generation_clusters = n_generation_clusters
        self.generation_max_new_tokens = generation_max_new_tokens
        self.generation_temperature = generation_temperature
        self.generation_log_every_n_steps = generation_log_every_n_steps
        self._gen_clusters: list[dict[str, Any]] | None = None

    def forward(self, **batch: Any) -> dict[str, torch.Tensor]:
        return self.model(**batch)

    def configure_optimizers(self) -> tuple[list[Optimizer], list[LRSchedulerConfig]]:
        trainer = self.trainer
        assert isinstance(trainer, Trainer), "Odin VAE harness expects the custom Trainer."  # nosec: B101
        optimizer_cls = getattr(optim, trainer.optimizer)
        optimizer = optimizer_cls(self.model.parameters(), lr=trainer.lr, **trainer.optimizer_kwargs)

        # ``estimated_stepping_batches`` / ``num_training_batches`` are unreliable for
        # length-less iterable datasets, so the total is derived from the per-epoch
        # batch limit (always an int for the endless train stream).
        limit = trainer.limit_train_batches
        per_epoch = float(limit) if isinstance(limit, int) and limit > 0 else float(trainer.num_training_batches)
        if per_epoch == float("inf") or per_epoch <= 0:
            raise ValueError("Cannot resolve the number of train batches per epoch for the LR schedule.")
        total_steps = one_cycle_total_steps(per_epoch, trainer.max_epochs or 1, trainer.accumulate_grad_batches)

        lr_scheduler_cfg: LRSchedulerConfig = cast(
            LRSchedulerConfig,
            {
                "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=trainer.lr,
                    total_steps=total_steps,
                    pct_start=trainer.lr_warmup_ratio,
                ),
                "interval": "step",
                "frequency": 1,
            },
        )
        return [optimizer], [lr_scheduler_cfg]

    def training_step(self, batch: dict[str, Any], batch_idx: int = 0) -> torch.Tensor:  # noqa: ARG002
        outputs = self.model(**batch)
        lr_val = self.lr_schedulers().get_last_lr()[0]  # type: ignore[attr-defined]
        self.log_dict(
            {
                "train_loss": outputs["loss"],
                "train_ce": outputs["ce"],
                "train_kl": outputs["kl"],
                "lr": lr_val if isinstance(lr_val, torch.Tensor) else float(lr_val),
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return cast(torch.Tensor, outputs["loss"])

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        outputs = self.model(**batch)
        self.log_dict(
            {"val_loss": outputs["loss"], "val_ce": outputs["ce"], "val_kl": outputs["kl"]},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return cast(torch.Tensor, outputs["loss"])

    # ------------------------------------------------------------------ #
    # Generation smoke check
    # ------------------------------------------------------------------ #
    def on_train_batch_end(self, *args: Any) -> None:
        del args
        trainer = self.trainer
        if trainer is None or not trainer.is_global_zero:
            return
        step = trainer.global_step
        if step > 0 and step % self.generation_log_every_n_steps == 0:
            self._log_generations("train")

    def on_validation_epoch_end(self) -> None:
        if self.trainer is None or not self.trainer.is_global_zero:
            return
        self._log_generations("val")

    def _load_generation_clusters(self) -> list[dict[str, Any]]:
        """Read a fixed set of cluster records from the first val shard."""
        if self._gen_clusters is not None or not self.val_shard_pattern:
            return self._gen_clusters or []
        files = [Path(p) for p in wds.shardlists.expand_urls(self.val_shard_pattern)]
        if not files:
            return []
        records: list[dict[str, Any]] = []
        with tarfile.open(str(files[0]), "r:gz") as tf:
            for member in tf.getmembers():
                if not member.name.endswith(".json") or len(records) >= self.n_generation_clusters:
                    continue
                fh = tf.extractfile(member)
                if fh is None:
                    continue
                record = json.loads(fh.read().decode("utf-8"))
                records.append({"key": member.name.rsplit(".", 1)[0], **record})
        self._gen_clusters = records
        return records

    def _encode_cluster(self, tags: list[str], cells: list[str], max_surfaces: int = 12) -> torch.Tensor:
        """Encode one cluster's surfaces (clean) and return its ``mu`` vector."""
        rows = list(zip(tags, cells))[:max_surfaces]
        ids = [self.tokenizer.encode(text, add_special_tokens=False)[:63] for _, text in rows]
        ids = [[self.tokenizer.convert_tokens_to_ids(f"{_TAG_PREFIX}{tag}]")] + row for tag, row in zip(tags, ids)]
        length = max(len(row) for row in ids)
        surf_ids = torch.full((len(ids), length), self.model.pad_token_id, dtype=torch.long)
        surf_mask = torch.zeros((len(ids), length), dtype=torch.long)
        for i, row in enumerate(ids):
            surf_ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
            surf_mask[i, : len(row)] = 1
        device = next(self.model.parameters()).device
        k_per_cluster = torch.full((1,), len(ids), dtype=torch.long)
        with torch.no_grad():
            mu, _ = self.model.encode(surf_ids.to(device), surf_mask.to(device), k_per_cluster=k_per_cluster)
        return mu[0]

    def _log_generations(self, kind: str) -> None:
        records = self._load_generation_clusters()
        if not records or self.logger is None:
            return
        tag_ids = {tag: self.tokenizer.convert_tokens_to_ids(f"{_TAG_PREFIX}{tag}]") for tag in SCRIPTS}
        lines: list[str] = []
        lines.append(f"# {kind} generation sample @ step {self.trainer.global_step}")
        lines.append(f"# generated: {datetime.datetime.now().isoformat()}")
        for record in records:
            lines.append(f"=== {record['key']} ===")
            mu = self._encode_cluster(record["tags"], record["cells"])
            lines.append("input surfaces:")
            for tag, cell in list(zip(record["tags"], record["cells"]))[:12]:
                lines.append(f"  [{tag}] {cell}")
            lines.append("generated:")
            for tag in SCRIPTS:
                ids = self.model.generate(
                    mu,
                    tag_ids[tag],
                    max_new_tokens=self.generation_max_new_tokens,
                    temperature=self.generation_temperature,
                )
                text = self.tokenizer.decode(ids, skip_special_tokens=True)
                lines.append(f"  [{tag}] {text}")
            # Unprimed decode: no alphabet information, the unknown-alphabet regime.
            ids = self.model.generate(
                mu,
                None,
                max_new_tokens=self.generation_max_new_tokens,
                temperature=self.generation_temperature,
            )
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            lines.append(f"  [unprimed] {text}")
        if self.logger.log_dir:
            out_dir = Path(self.logger.log_dir) / "generations"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{kind}-step{self.trainer.global_step}.txt"
            out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def on_fit_start(self) -> None:
        if self.trainer is not None and self.trainer.is_global_zero:
            self._load_generation_clusters()

    def on_fit_end(self) -> None:
        if self.trainer is None or not self.trainer.is_global_zero:
            return
        self._log_generations("final")
