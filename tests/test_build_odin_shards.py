"""Tests for the webdataset shard builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from _build_loader import get_builder
from odin_vae.data.grandwds import GrandWebDataset

_builder = get_builder()

SCRIPTS = ["la", "cy", "gk", "ab", "cn", "jp", "kr", "dv", "hb", "th", "gg", "am"]


def make_line(i: int) -> str:
    return "\t".join(f"{tag}{{value-{i}-{tag}}}" for tag in SCRIPTS)


def test_parse_line_round_trip() -> None:
    line = make_line(7)
    tags, cells = _builder.parse_line(line)
    assert tags == SCRIPTS
    assert cells == [f"value-7-{tag}" for tag in SCRIPTS]


@pytest.mark.parametrize(
    "line",
    [
        "",
        "la{a}\tzz{b}",  # unknown tag
        "la{a}\tla{",  # unterminated
        "la{}\tla{b}",  # empty value
        "la{a}",  # no la? this one has la... use a no-la line instead
    ],
)
def test_parse_line_rejects(line: str) -> None:
    if line == "la{a}":
        return  # valid: has la
    assert _builder.parse_line(line) is None


def test_parse_line_requires_la() -> None:
    assert _builder.parse_line("cn{a}\tcy{b}") is None


def test_val_index_set_properties() -> None:
    total, val_size = 10_000, 500
    indices = _builder.val_index_set(total, val_size)
    assert len(indices) == val_size
    assert min(indices) == 0
    assert max(indices) < total
    # Regular cadence: gaps are val_size-adjacent (here ~20).
    gaps = [b - a for a, b in zip(sorted(indices), sorted(indices)[1:])]
    assert all(g in (19, 20) for g in gaps)
    assert _builder.val_index_set(10, 0) == set()


def test_minimal_build_round_trip(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    n = 600
    corpus.write_text("\n".join(make_line(i) for i in range(n)) + "\n", encoding="utf-8")
    out = tmp_path / "wds"
    _builder.main(
        [
            "--input",
            str(corpus),
            "--output",
            str(out),
            "--workers",
            "2",
            "--shard-size",
            "256",
            "--val-size",
            "100",
            "--limit",
            str(n),
        ]
    )

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["n_train"] == 500
    assert manifest["n_val"] == 100
    assert manifest["n_train_shards"] == 2
    assert manifest["n_val_shards"] == 1

    # Read the train shards back through the reader pipeline.
    dataset = GrandWebDataset(f"{out}/{manifest['train_pattern']}", seed=123, loop_back=False)
    records = {item["__key__"]: json.loads(item["json"]) for item in dataset}
    assert len(records) == 500
    rec = records["cluster-000000000"]
    assert rec["tags"] == SCRIPTS
    assert len(rec["cells"]) == 12
    # The train lines are the corpus lines minus the val slots (k*6).
    train_indices = [i for i in range(n) if i not in {(k * n) // 100 for k in range(100)}]
    assert {rec["cells"][0] for rec in records.values()} == {f"value-{i}-la" for i in train_indices}
    # A record from the second shard (train line 256).
    rec256 = records["cluster-000000256"]
    assert rec256["cells"][0].startswith("value-")

    # Val shards: namespaced keys, disjoint from train.
    val_dataset = GrandWebDataset(f"{out}/{manifest['val_pattern']}", seed=123, loop_back=False)
    val_records = {item["__key__"]: json.loads(item["json"]) for item in val_dataset}
    assert len(val_records) == 100
    assert all(k.startswith("val-") for k in val_records)
    assert set(val_records).isdisjoint(records)
    # val slot k is corpus line k*6: the first val record is corpus line 0.
    assert val_records["val-000000000"]["cells"] == [f"value-0-{tag}" for tag in SCRIPTS]

    # The letter-frequency table covers the corruptible scripts.
    freq = json.loads((out / "char_frequencies.json").read_text())
    assert "la" in freq and "gg" in freq
    assert "cn" not in freq and "jp" not in freq and "kr" not in freq
    # 'v', 'a', 'l', 'u', 'e' all appear in every cell.
    la_letters = {ch for ch, _ in freq["la"]}
    assert set("value") <= la_letters


def test_build_with_explicit_val_input(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    n = 8
    corpus.write_text("\n".join(make_line(i) for i in range(n)) + "\n", encoding="utf-8")
    # Val = one line that also appears in the corpus (leakage guard) plus one
    # external line not present in the corpus at all.
    val_file = tmp_path / "val.txt"
    external = "\t".join(f"{tag}{{ext-{tag}}}" for tag in SCRIPTS)
    val_file.write_text(make_line(3) + "\n" + external + "\n", encoding="utf-8")
    out = tmp_path / "wds"
    _builder.main(
        [
            "--input",
            str(corpus),
            "--output",
            str(out),
            "--workers",
            "1",
            "--shard-size",
            "4",
            "--val-input",
            str(val_file),
        ]
    )

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["val_input"] == str(val_file)
    assert manifest["n_val"] == 2
    assert manifest["n_train"] == 7  # 8 corpus lines minus the one overlapping val line
    assert manifest["n_train_shards"] == 2
    assert manifest["n_val_shards"] == 1

    records = {
        item["__key__"]: json.loads(item["json"])
        for item in GrandWebDataset(f"{out}/{manifest['train_pattern']}", seed=123, loop_back=False)
    }
    assert len(records) == 7
    assert all(rec["cells"][0] != "value-3-la" for rec in records.values())  # val line never trained on

    val_records = [
        json.loads(item["json"])
        for item in GrandWebDataset(f"{out}/{manifest['val_pattern']}", seed=123, loop_back=False)
    ]
    # File order preserved: the corpus line first, then the external line.
    assert [rec["cells"][0] for rec in val_records] == ["value-3-la", "ext-la"]


def test_build_with_val_shard_size(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    n = 12
    corpus.write_text("\n".join(make_line(i) for i in range(n)) + "\n", encoding="utf-8")
    # Explicit test set: 4 lines, none present in the corpus (no leakage interaction).
    test_file = tmp_path / "test.txt"
    test_file.write_text(
        "\n".join("\t".join(f"{tag}{{t-{i}-{tag}}}" for tag in SCRIPTS) for i in range(4)) + "\n", encoding="utf-8"
    )
    out = tmp_path / "wds"
    _builder.main(
        [
            "--input",
            str(corpus),
            "--output",
            str(out),
            "--workers",
            "1",
            "--shard-size",
            "8",
            "--val-size",
            "5",
            "--val-shard-size",
            "2",
            "--test-input",
            str(test_file),
            "--test-shard-size",
            "3",
        ]
    )
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["shard_size"] == 8
    assert manifest["val_shard_size"] == 2
    assert manifest["n_val"] == 5
    assert manifest["n_val_shards"] == 3  # 2 + 2 + 1
    assert manifest["n_train"] == 7
    assert manifest["n_train_shards"] == 1
    assert manifest["test_shard_size"] == 3
    assert manifest["n_test"] == 4
    assert manifest["n_test_shards"] == 2  # 3 + 1


def test_build_counts_malformed(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    lines = [make_line(0), "garbage-line", make_line(1)]
    corpus.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = tmp_path / "wds"
    _builder.main(
        ["--input", str(corpus), "--output", str(out), "--workers", "1", "--shard-size", "10", "--val-size", "1"]
    )
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["n_total"] == 3
    assert manifest["n_train"] + manifest["n_val"] == 2
