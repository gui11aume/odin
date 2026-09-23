"""Integration tests: built shards -> GrandWebDataset -> collator -> model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightning.pytorch as pl
import pytest
import torch

from _build_loader import get_builder
from odin_vae.augment import LetterAugmenter
from odin_vae.config_classes import ConfigForModel
from odin_vae.data.collators import OdinVAECollator
from odin_vae.data.data_modules import OdinVAEDataModule
from odin_vae.data.grandwds import GrandShardList, GrandWebDataset
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
    dataset = GrandWebDataset(f"{shard_root}/{manifest['train_pattern']}", seed=123, loop_back=False)
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
    dataset = GrandWebDataset(pattern, seed=123, loop_back=False)
    dataset.set_progress(0, 2)  # skip the first two shards
    urls_seen: list[str] = []
    for item in dataset:
        url = item["__url__"]
        if not urls_seen or urls_seen[-1] != url:
            urls_seen.append(url)
    assert urls_seen == epoch0[2:]


def test_dataset_progress_round_trip(shard_root: Path) -> None:
    import itertools
    import random

    import webdataset as wds

    manifest = json.loads((shard_root / "manifest.json").read_text())
    pattern = f"{shard_root}/{manifest['train_pattern']}"

    def shard_stream(dataset: GrandWebDataset) -> list[str]:
        seen: list[str] = []
        for item in dataset:
            url = item["__url__"]
            if not seen or seen[-1] != url:
                seen.append(url)
        return seen

    src = GrandWebDataset(pattern, seed=123, loop_back=False)
    src.set_progress(1, 3)
    dst = GrandWebDataset(pattern, seed=123, loop_back=False)
    assert dst.seed == 123
    dst.set_progress(*src.progress())
    assert dst.progress() == (1, 3)
    assert shard_stream(dst) == shard_stream(src)
    # Endless streams resume at the offset of the epoch permutation.
    perm = wds.shardlists.expand_urls(pattern)
    random.Random(123).shuffle(perm)
    src_e = GrandWebDataset(pattern, seed=123, loop_back=True)
    src_e.set_progress(0, 1)
    head = [d["url"] for d in itertools.islice(src_e.shardlist, 3)]
    assert head == perm[1:4]


def make_auto_loader(shard_root: Path, **loader_kwargs):
    from odin_vae.data.data_loaders import DataLoaderWithAutoCheckpoint

    manifest = json.loads((shard_root / "manifest.json").read_text())
    pattern = f"{shard_root}/{manifest['train_pattern']}"
    dataset = GrandWebDataset(pattern, seed=123, loop_back=False)
    kwargs = dict(batch_size=8, num_workers=0, **loader_kwargs)
    return DataLoaderWithAutoCheckpoint(dataset=dataset, **kwargs), dataset


def test_loader_state_round_trip(shard_root: Path) -> None:
    # Save-side: state_dict() resolves progress and stores samples (exact).
    loader, dataset = make_auto_loader(shard_root, progress_fn=lambda: (0, 512), n_instances_per_shard=SHARD_SIZE)
    state = loader.state_dict()
    assert state == {"seed": 123, "processed_epochs": 0, "processed_samples": 512}
    assert dataset.progress() == (0, 2)
    # Load-side: the state drives a fresh dataset's offset.
    loader2, dataset2 = make_auto_loader(shard_root, n_instances_per_shard=SHARD_SIZE)
    loader2.load_state_dict(state)
    assert dataset2.progress() == (0, 2)
    # A progress-less (val-style) loader rejects progress-carrying state.
    loader_val, _ = make_auto_loader(shard_root)
    with pytest.raises(ValueError, match="n_instances_per_shard"):
        loader_val.load_state_dict(state)
    # No trainer attached: state_dict() preserves the last known progress.
    assert loader2.state_dict() == state


def test_loader_state_rejects_bad_boundaries(shard_root: Path) -> None:
    import pytest

    loader, _ = make_auto_loader(shard_root, n_instances_per_shard=SHARD_SIZE)
    with pytest.raises(ValueError, match="non-shard-boundary"):
        loader.load_state_dict({"seed": 123, "processed_epochs": 0, "processed_samples": SHARD_SIZE + 8})
    with pytest.raises(ValueError, match="seed"):
        loader.load_state_dict({"seed": 999, "processed_epochs": 0, "processed_samples": SHARD_SIZE})


def test_loader_legacy_seed_only_bootstraps_once(shard_root: Path) -> None:
    # Legacy checkpoints carry only the seed: the next __iter__ derives the
    # progress from the Trainer exactly once, then the flag self-extinguishes.
    loader, dataset = make_auto_loader(shard_root, progress_fn=lambda: (0, 512), n_instances_per_shard=SHARD_SIZE)
    loader.load_state_dict({"seed": 123})
    assert dataset.progress() == (0, 0)
    iter(loader)
    assert dataset.progress() == (0, 2)
    # Second iteration: no re-bootstrap even if progress would differ.
    loader._progress_fn = lambda: (1, 0)
    iter(loader)
    assert dataset.progress() == (0, 2)


def test_data_progress_boundary_normalization(shard_root: Path) -> None:
    class _NS:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    dm = make_datamodule(shard_root, make_collator(shard_root), batch_size=8)
    dm.trainer = None
    assert dm._data_progress() is None
    batch_progress = _NS(current=_NS(processed=32), is_last_batch=False)
    epoch_progress = _NS(current=_NS(processed=1, ready=2))
    dm.trainer = _NS(fit_loop=_NS(epoch_progress=epoch_progress, epoch_loop=_NS(batch_progress=batch_progress)))
    assert dm._data_progress() == (1, 32 * 8)
    # Boundary: saved on the last batch -> normalize to the next epoch start.
    batch_progress.is_last_batch = True
    assert dm._data_progress() == (2, 0)


class _CountingModule(pl.LightningModule):
    """Lossless stand-in module for checkpoint round-trip tests."""

    def __init__(self) -> None:
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def training_step(self, batch: dict, _: int) -> torch.Tensor:
        return self.p.sum()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD([self.p], lr=0.1)


def test_checkpoint_resumes_data_offset(shard_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A real Lightning save/restore round-trip drives the shard offset.

    32 batches x batch_size 8 = 256 = exactly one shard, so checkpoints land
    on shard boundaries. The resumed run must continue from the checkpoint's
    shard offset of the same epoch permutation.
    """
    import random

    import webdataset as wds
    from lightning.pytorch.callbacks import ModelCheckpoint

    manifest = json.loads((shard_root / "manifest.json").read_text())
    pattern = f"{shard_root}/{manifest['train_pattern']}"
    perm = wds.shardlists.expand_urls(pattern)
    random.Random(123).shuffle(perm)

    # Observe which shards the pipeline actually consumes (main process:
    # num_workers=0).
    consumed: list[list[str]] = []
    current: list[str] = []
    original = GrandShardList.__iter__

    def probed(self):  # type: ignore[no-untyped-def]
        for d in original(self):
            if not current or current[-1] != d["url"]:
                current.append(d["url"])
            yield d

    monkeypatch.setattr(GrandShardList, "__iter__", probed)

    def make_trainer(ckpt_dir: Path, max_steps: int) -> pl.Trainer:
        return pl.Trainer(
            max_steps=max_steps,
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=True,
            callbacks=[
                ModelCheckpoint(
                    dirpath=str(ckpt_dir),
                    every_n_train_steps=32,
                    save_top_k=-1,
                    filename="step{step:04d}",
                    auto_insert_metric_name=False,
                )
            ],
            limit_val_batches=0,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

    # Run 1: 64 steps = 2 shards -> checkpoints at step 32 (1 shard) and 64.
    dm = make_datamodule(shard_root, make_collator(shard_root), batch_size=8)
    make_trainer(shard_root / "ckpts", max_steps=64).fit(_CountingModule(), datamodule=dm)
    consumed.append(current.copy())
    # The fetcher prefetches one batch into the next shard; the fully
    # consumed prefix must be the head of the epoch permutation.
    assert consumed[0][:2] == perm[:2]

    ckpt_path = shard_root / "ckpts" / "step0032.ckpt"
    assert ckpt_path.is_file()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # nosec: B614  # our own checkpoint
    state = ckpt["loops"]["fit_loop"]["state_dict"]["combined_loader"][0]
    assert state == {"seed": 123, "processed_epochs": 0, "processed_samples": SHARD_SIZE}

    # Run 2: resume from the 1-shard checkpoint -> continue at perm[1].
    current.clear()
    dm2 = make_datamodule(shard_root, make_collator(shard_root), batch_size=8)
    dm2.setup("fit")
    dataset2 = dm2.datasets["train"]  # teardown clears the cache after fit
    make_trainer(shard_root / "ckpts2", max_steps=64).fit(_CountingModule(), datamodule=dm2, ckpt_path=ckpt_path)
    consumed.append(current.copy())
    assert consumed[1][0] == perm[1]  # resumed at the checkpoint's shard offset
    # The run consumed 32 more batches = 1 shard, starting from the offset.
    assert dataset2.progress() == (0, 2)

    # Run 3: legacy checkpoint (seed-only state) bootstraps from the restored
    # Trainer counters and lands on the same offset.
    legacy_path = shard_root / "ckpts" / "step0032.legacy.ckpt"
    del state["processed_epochs"], state["processed_samples"]
    ckpt["loops"]["fit_loop"]["state_dict"]["combined_loader"][0] = dict(state)
    torch.save(ckpt, legacy_path)
    current.clear()
    dm3 = make_datamodule(shard_root, make_collator(shard_root), batch_size=8)
    make_trainer(shard_root / "ckpts3", max_steps=64).fit(_CountingModule(), datamodule=dm3, ckpt_path=legacy_path)
    consumed.append(current.copy())
    assert consumed[2][0] == perm[1]  # legacy bootstrap landed on the same offset


def test_val_split_finite(shard_root: Path) -> None:
    manifest = json.loads((shard_root / "manifest.json").read_text())
    dataset = GrandWebDataset(f"{shard_root}/{manifest['val_pattern']}", seed=123, loop_back=False)
    keys = [item["__key__"] for item in dataset]
    assert len(keys) == manifest["n_val"]
    # Disjoint from train by construction of the build.
    train_dataset = GrandWebDataset(f"{shard_root}/{manifest['train_pattern']}", seed=123, loop_back=False)
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
    k0 = int(batch["k_per_cluster"][0])
    with torch.no_grad():
        mu, _ = model.encode(
            batch["surf_ids"][:k0].to(device),
            batch["surf_mask"][:k0].to(device),
            k_per_cluster=batch["k_per_cluster"][:1].to(device),
        )
    tag_id = tokenizer.convert_tokens_to_ids("[gk]")
    ids = model.generate(mu[0], tag_id, max_new_tokens=8)
    ids_unprimed = model.generate(mu[0], None, max_new_tokens=8)
    assert len(ids) <= 8
    assert len(ids_unprimed) <= 8
    text = tokenizer.decode(ids, skip_special_tokens=True)
    assert isinstance(text, str)
    sys.stdout.write(f"generated greek surface: {text!r}\n")
