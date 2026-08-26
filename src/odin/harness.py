"""PyTorch Lightning training utilities for masked language modelling."""

from __future__ import annotations

import datetime
import os
import socket
from typing import Any, cast

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import LRSchedulerConfig
from torch import optim
from torch.optim.optimizer import Optimizer
from transformers import PreTrainedModel


class Trainer(pl.Trainer):
    """Lightning ``Trainer`` with optimizer hyper-parameters injected like the Arena template."""

    overfit_batches: int | float
    reload_dataloaders_every_n_epochs: int
    log_every_n_steps: int

    def __init__(self, **kwargs: Any):
        self.lr: float = float(kwargs.pop("lr", 5e-5))
        self.lr_warmup_ratio: float = float(kwargs.pop("lr_warmup_ratio", 0.1))
        self.optimizer: str = str(kwargs.pop("optimizer", "AdamW"))
        self.optimizer_kwargs: dict[str, Any] = dict(kwargs.pop("optimizer_kwargs", {}))
        super().__init__(**kwargs)


class MLMLightningHarness(pl.LightningModule):
    """Wraps ``AutoModelForMaskedLM`` and logs MLM losses."""

    def __init__(self, model: PreTrainedModel):
        super().__init__()
        self.model: PreTrainedModel = model

    def forward(self, **batch: torch.Tensor):
        return self.model(**batch)

    def configure_optimizers(self) -> tuple[list[Optimizer], list[LRSchedulerConfig]]:
        trainer = self.trainer
        assert isinstance(trainer, Trainer), "Odin harness expects the custom Trainer."
        optimizer_cls = getattr(optim, trainer.optimizer)
        optimizer = optimizer_cls(self.model.parameters(), lr=trainer.lr, **trainer.optimizer_kwargs)

        if trainer.estimated_stepping_batches is None:
            trainer.fit_loop.setup_data()  # pragma: no cover - lightning-specific wiring
        stepping_batches = trainer.estimated_stepping_batches
        assert stepping_batches is not None

        lr_scheduler_cfg: LRSchedulerConfig = cast(
            LRSchedulerConfig,
            {
                "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=trainer.lr,
                    total_steps=max(1, int(stepping_batches)),
                    pct_start=trainer.lr_warmup_ratio,
                ),
                "interval": "step",
                "frequency": 1,
            },
        )

        return [optimizer], [lr_scheduler_cfg]

    def on_fit_start(self) -> None:
        """Log bookkeeping metadata."""
        self._log_metadata_start()

    def on_fit_end(self) -> None:
        """Finalize metadata."""
        self._log_metadata_end()

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int = 0) -> torch.Tensor:  # noqa: ARG002
        outputs = self.model(**batch)
        loss = outputs["loss"]
        lr_val = self.lr_schedulers().get_last_lr()[0]  # type: ignore[attr-defined]
        payload: dict[str, torch.Tensor | float] = {"loss": loss}
        payload["lr"] = lr_val if isinstance(lr_val, torch.Tensor) else float(lr_val)
        self.log_dict(payload, on_step=True, prog_bar=True)
        return cast(torch.Tensor, loss)

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        outputs = self.model(**batch)
        loss = outputs["loss"]
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return cast(torch.Tensor, loss)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        outputs = self.model(**batch)
        loss = outputs["loss"]
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return cast(torch.Tensor, loss)

    def _log_metadata_start(self) -> None:
        if self.trainer is None or not self.trainer.is_global_zero or self.logger is None:
            return

        if bool(self.hparams):
            self.logger.log_hyperparams(dict(self.hparams))

        dl = getattr(self.trainer, "train_dataloader", None)
        if callable(dl):
            dl = dl()
        if dl is None:
            self.trainer.fit_loop.setup_data()
            dl = getattr(self.trainer, "train_dataloader", None)
            if callable(dl):
                dl = dl()

        if dl is not None:
            self.logger.log_hyperparams(
                {
                    "batch_size": getattr(dl, "batch_size", None),
                    "num_workers": getattr(dl, "num_workers", None),
                }
            )

        assert isinstance(self.trainer, Trainer)
        trainer_params: dict[str, Any] = {
            "lr": self.trainer.lr,
            "lr_warmup_ratio": self.trainer.lr_warmup_ratio,
            "optimizer": self.trainer.optimizer,
            "accumulate_grad_batches": self.trainer.accumulate_grad_batches,
            "precision": self.trainer.precision,
            "accelerator": str(self.trainer.accelerator),
            "strategy": type(self.trainer.strategy).__name__,
            "devices": self.trainer.num_devices,
            "hostname": socket.gethostname(),
        }
        self.logger.log_hyperparams(trainer_params)

        log_dir = self.logger.log_dir
        if log_dir:
            summary_path = os.path.join(log_dir, "model_summary.txt")
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            with open(summary_path, "w", encoding="utf-8") as handle:
                name = getattr(self.model, "name_or_path", self.model.__class__.__name__)
                handle.write(str(name))

        self.logger.log_hyperparams({"_time_start": datetime.datetime.now().isoformat(), "_reached_fit_end": False})

    def _log_metadata_end(self) -> None:
        if self.logger is None or self.trainer is None:
            return
        self.logger.log_hyperparams(
            {
                "_time_end": datetime.datetime.now().isoformat(),
                "_total_steps": int(self.trainer.global_step),
                "_reached_fit_end": True,
            }
        )
