"""Tests for the GrandWebDataset shard stream (order, resume, DDP split)."""

from __future__ import annotations

from pathlib import Path

import pytest
import webdataset as wds

from odin_vae.data.grandwds import GrandShardList, GrandWebDataset

SHARDS = ["shard-000000.tar.gz", "shard-000001.tar.gz", "shard-000002.tar.gz"]
# Pinned permutations of SHARDS for seed=123 (random.Random(123 + epoch)).
EPOCH0 = ["shard-000002.tar.gz", "shard-000001.tar.gz", "shard-000000.tar.gz"]
EPOCH1 = ["shard-000002.tar.gz", "shard-000000.tar.gz", "shard-000001.tar.gz"]


def build_shards(tmp_path: Path) -> list[str]:
    """Three shards of two samples each; shard i holds keys x{i}0, x{i}1."""
    paths = []
    for i, name in enumerate(SHARDS):
        path = tmp_path / name
        with wds.TarWriter(str(path)) as sink:
            for j in range(2):
                sink.write({"__key__": f"x{i}{j}", "json": b'{"tags":["la"],"cells":["n"]}'})
        paths.append(str(path))
    return paths


def keys_of(dataset, limit: int | None = None) -> list[str]:
    out = []
    for item in dataset:
        out.append(item["__key__"])
        if limit is not None and len(out) >= limit:
            break
    return out


def test_shardlist_permutation_pinned() -> None:
    shardlist = GrandShardList(SHARDS, seed=123, is_endless=False)
    assert [d["url"] for d in shardlist] == EPOCH0


def test_shardlist_resume_offset() -> None:
    shardlist = GrandShardList(SHARDS, seed=123, is_endless=False)
    shardlist.set_progress(0, 2)
    assert [d["url"] for d in shardlist] == EPOCH0[2:]


def test_full_pipeline_order(tmp_path: Path) -> None:
    paths = build_shards(tmp_path)
    dataset = GrandWebDataset(paths, seed=123, is_endless=False)
    assert keys_of(dataset) == ["x20", "x21", "x10", "x11", "x00", "x01"]


def test_endless_cycles(tmp_path: Path) -> None:
    paths = build_shards(tmp_path)
    dataset = GrandWebDataset(paths, seed=123, is_endless=True)
    first_cycle = ["x20", "x21", "x10", "x11", "x00", "x01"]
    assert keys_of(dataset, limit=12) == first_cycle + first_cycle


def test_resume_skips_processed_shards(tmp_path: Path) -> None:
    paths = build_shards(tmp_path)
    dataset = GrandWebDataset(paths, seed=123, is_endless=False)
    dataset.set_progress(0, 2)
    assert keys_of(dataset) == ["x00", "x01"]


def test_ddp_split_disjoint(tmp_path: Path) -> None:
    paths = build_shards(tmp_path)
    # Epoch-0 permutation: shard-2, shard-1, shard-0. Stride 2:
    # rank 0 -> [shard-2, shard-0], rank 1 -> [shard-1].
    for rank, expected in ((0, ["x20", "x21", "x00", "x01"]), (1, ["x10", "x11"])):
        dataset = GrandWebDataset(paths, seed=123, is_endless=False)
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("RANK", str(rank))
            mp.setenv("WORLD_SIZE", "2")
            assert keys_of(dataset) == expected


def test_seed_and_progress_api(tmp_path: Path) -> None:
    paths = build_shards(tmp_path)
    dataset = GrandWebDataset(paths, seed=123, is_endless=True)
    assert dataset.seed == 123
    assert dataset.progress() == (0, 0)
    dataset.set_progress(1, 2)
    assert dataset.progress() == (1, 2)
    # The offset applies to the resumed epoch's permutation: skipping two
    # shards of epoch 1 (shard-2, shard-0) leaves shard-1.
    shardlist = GrandShardList(paths, seed=123, is_endless=False)
    shardlist.set_progress(1, 2)
    assert [d["url"] for d in shardlist] == [paths[SHARDS.index(EPOCH1[2])]]


def test_no_shards_raises(tmp_path: Path) -> None:
    dataset = GrandWebDataset([str(tmp_path / "missing-0.tar.gz")], seed=123, is_endless=False)
    with pytest.raises(Exception):
        list(dataset)


def test_expand_urls_in_prepare_data_pattern(tmp_path: Path) -> None:
    build_shards(tmp_path)
    pattern = str(tmp_path / "shard-{000000..000002}.tar.gz")
    files = wds.shardlists.expand_urls(pattern)
    assert files == [str(tmp_path / name) for name in SHARDS]
