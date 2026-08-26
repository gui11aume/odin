"""Configuration validation helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from odin.config_classes import ConfigForRoot


def test_fixture_config_roundtrip(tmp_path: Path, repo_root_path: Path) -> None:
    fixture_dir = repo_root_path / "tests/fixtures"
    train_txt = fixture_dir / "mlm_lines_train.txt"
    val_txt = fixture_dir / "mlm_lines_val.txt"
    yaml_payload = {
        "datamodule": {
            "train_text_path": str(train_txt),
            "val_text_path": str(val_txt),
            "dataloader": {
                "batch_size": 2,
                "num_workers": 0,
                "drop_last_batch": False,
                "persistent_workers": False,
                "pin_memory": False,
            },
            "collator": {"max_seq_length": 48, "mlm_probability": 0.25},
        },
        "models": {
            "tokenizer": {},
            "modern_bert": {
                "pretrained_model_name_or_path": "allenai/modernbert-base",
                "trust_remote_code": False,
            },
        },
        "training": {
            "strategy": "auto",
            "accelerator": "cpu",
            "devices": 1,
            "precision": "32-true",
            "max_epochs": 1,
            "max_steps": -1,
            "limit_train_batches": 2,
            "limit_val_batches": 2,
            "log_every_n_steps": 10,
            "num_sanity_val_steps": 0,
            "val_check_interval": 1.0,
            "checkpoint_path": None,
        },
    }

    dumped = yaml.safe_dump(yaml_payload)
    (tmp_path / "cfg.yaml").write_text(dumped, encoding="utf-8")
    loaded = yaml.safe_load((tmp_path / "cfg.yaml").read_text(encoding="utf-8"))

    validated = ConfigForRoot.from_mapping(loaded)
    assert validated.datamodule.collator.max_seq_length == 48


def test_train_path_must_exist(tmp_path: Path) -> None:
    missing_yaml = {
        "datamodule": {
            "train_text_path": str(tmp_path / "nope.txt"),
            "dataloader": {"batch_size": 4},
        },
        "models": {
            "tokenizer": {},
            "modern_bert": {"pretrained_model_name_or_path": "allenai/modernbert-base"},
        },
        "training": {"accelerator": "cpu", "devices": 1, "strategy": "auto"},
    }

    with pytest.raises(ValueError):
        ConfigForRoot.from_mapping(missing_yaml)
