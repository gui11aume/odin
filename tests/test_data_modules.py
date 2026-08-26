"""Integration tests: built shards -> GrandWebDataset -> collator -> model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from _build_loader import get_builder
from odin_vae.augment import LetterAugmenter
from odin_vae.config_classes import ConfigForModel
from odin_vae.data.collators import OdinVAECollator
from odin_vae.data.data_modules import OdinVAEDataModule
from odin_vae.data.grandwds import GrandWebDataset
from odin_vae.model import OdinModel

_builder = get_builder()

TOKENIZER_PATH = "/mnt/nvme1/odin_tokenizer"
SHARD_SIZE = 256
N_CLUSTERS = 1000
VAL_SIZE = 100


@pytest.fixture(scope="module")
def shard_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny but real webdataset build (1000 clusters, 8 workers of 1)."""
    root = tmp_path_factory.mktemp("odin_wds")
    corpus = Path("/mnt/nvme1/odin_train_set.clean.txt.gz")
    _builder.main(
        [
            "--input",
            str(corpus),
            "--output",
            str(root),
            "--workers",
            "2",
            "--shard-size",
            str(SHARD_SIZE),
            "--val-size",
            str(VAL_SIZE),
            "--limit",
            str(N_CLUSTERS),
        ]
    )
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["n_train"] == N_CLUSTERS - VAL_SIZE
    return root


def make_collator(shard_root: Path) -> OdinVAECollator:
    from transformers import PreTrainedTokenizerFast

    freq = json.loads((shard_root / "char_frequencies.json").read_text())
    augmenter = LetterAugmenter(rate=0.015, letter_frequencies=freq)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)  # nosec: B615  # local dir, not the Hub
    return OdinVAECollator(tokenizer, augmenter, k_input=4, k_latin_target=4, k_non_latin_target=2)


def make_model(device: str = "cpu") -> OdinModel:
    config = ConfigForModel(
        tokenizer_path=TOKENIZER_PATH,
        hidden_size=32,
        attention_heads=2,
        intermediate_size=64,
        encoder_layers=2,
        decoder_layers=3,
        local_attention=16,
        max_position_embeddings=48,
        decoder="modernbert",
    )
    return OdinModel(config, vocab_size=16384, pad_token_id=0, bos_token_id=2, eos_token_id=3).to(device)


def make_datamodule(shard_root: Path, collator, batch_size: int = 8) -> OdinVAEDataModule:
    manifest = json.loads((shard_root / "manifest.json").read_text())
    config = {
        "data_root": str(shard_root),
        "splits": {
            "train": {
                "dataset": {"pattern": manifest["train_pattern"], "n_instances_per_shard": SHARD_SIZE},
                "dataloader": {"batch_size": batch_size, "num_workers": 0},
            },
            "val": {
                "dataset": {"pattern": manifest["val_pattern"], "n_instances_per_shard": SHARD_SIZE},
                "dataloader": {"batch_size": batch_size, "num_workers": 0},
            },
        },
    }
    return OdinVAEDataModule(config=config, collator=collator, seed=123)


def test_reader_round_trip(shard_root: Path) -> None:
    manifest = json.loads((shard_root / "manifest.json").read_text())
    dataset = GrandWebDataset(f"{shard_root}/{manifest['train_pattern']}", seed=123, is_endless=False)
    keys = [item["__key__"] for item in dataset]
    assert len(keys) == manifest["n_train"]
    assert keys[0].startswith("cluster-")
    assert len(set(keys)) == len(keys)


def test_full_forward_pass(shard_root: Path, device: str = "cpu") -> None:
    collator = make_collator(shard_root)
    datamodule = make_datamodule(shard_root, collator, batch_size=8)
    datamodule.setup("fit")
    loader = datamodule.train_dataloader()
    assert loader is not None
    batch = next(iter(loader))
    model = make_model(device)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    out = model(**batch)
    assert torch.isfinite(out["loss"])
    assert out["loss"].item() > 0


def test_resume_skips_processed_shards(shard_root: Path) -> None:
    import random

    import webdataset as wds

    manifest = json.loads((shard_root / "manifest.json").read_text())
    pattern = f"{shard_root}/{manifest['train_pattern']}"
    urls = wds.shardlists.expand_urls(pattern)
    epoch0 = urls[:]
    random.Random(123 + 0).shuffle(epoch0)
    dataset = GrandWebDataset(pattern, seed=123, is_endless=False)
    dataset._set_shared_progress_from_main(0, 2)  # skip the first two shards
    urls_seen: list[str] = []
    for item in dataset:
        url = item["__url__"]
        if not urls_seen or urls_seen[-1] != url:
            urls_seen.append(url)
    assert urls_seen == epoch0[2:]


def test_val_split_finite(shard_root: Path) -> None:
    manifest = json.loads((shard_root / "manifest.json").read_text())
    dataset = GrandWebDataset(f"{shard_root}/{manifest['val_pattern']}", seed=123, is_endless=False)
    keys = [item["__key__"] for item in dataset]
    assert len(keys) == manifest["n_val"]
    # Disjoint from train by construction of the build.
    train_dataset = GrandWebDataset(f"{shard_root}/{manifest['train_pattern']}", seed=123, is_endless=False)
    train_keys = {item["__key__"] for item in train_dataset}
    assert train_keys.isdisjoint(keys)


def test_generate_from_real_cluster(shard_root: Path, device: str = "cpu") -> None:
    from transformers import PreTrainedTokenizerFast

    collator = make_collator(shard_root)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)  # nosec: B615  # local dir, not the Hub
    datamodule = make_datamodule(shard_root, collator, batch_size=8)
    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))
    model = make_model(device)
    model.eval()
    b = int(batch["n_clusters"])
    k_in = batch["surf_ids"].shape[0] // b
    with torch.no_grad():
        mu, _ = model.encode(batch["surf_ids"][:k_in].to(device), batch["surf_mask"][:k_in].to(device), k=k_in)
    tag_id = tokenizer.convert_tokens_to_ids("[gk]")
    ids = model.generate(mu[0], tag_id, max_new_tokens=8)
    assert len(ids) <= 8
    text = tokenizer.decode(ids, skip_special_tokens=True)
    assert isinstance(text, str)
    sys.stdout.write(f"generated greek surface: {text!r}\n")
