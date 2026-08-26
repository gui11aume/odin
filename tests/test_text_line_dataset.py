"""Tests for the line-oriented corpus dataset."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from odin.data.text_line_dataset import TextLineDataset


def test_reads_plain_text_lines(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("alpha beta\n gamma \n\t\n", encoding="utf-8")
    dataset = TextLineDataset(corpus)
    assert list(dataset.lines) == ["alpha beta", "gamma"]


def test_reads_gzipped_utf8_lines(tmp_path: Path) -> None:
    gz_path = tmp_path / "corpus.gz"
    payload = "\nhello world\n encore\n".encode("utf-8")
    gz_path.write_bytes(gzip.compress(payload))
    dataset = TextLineDataset(gz_path)
    assert dataset[0] == "hello world"
    assert len(dataset) == 2


def test_empty_raises(tmp_path: Path) -> None:
    corpus = tmp_path / "blank.txt"
    corpus.write_text("\n\n\t\n", encoding="utf-8")
    with pytest.raises(ValueError):
        TextLineDataset(corpus)


def test_fixture_lines_are_discoverable(repo_root_path: Path) -> None:
    dataset = TextLineDataset(repo_root_path / "tests/fixtures/mlm_lines_train.txt")
    assert len(dataset) >= 1
