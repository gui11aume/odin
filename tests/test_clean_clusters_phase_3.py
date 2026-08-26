"""Tests for the phase-3 cluster corpus cleaner."""

from __future__ import annotations

import gzip
import importlib.util
from pathlib import Path

import pytest

_RUNNER = Path(__file__).resolve().parents[1] / "runners" / "clean_clusters_phase_3.py"
_spec = importlib.util.spec_from_file_location("clean_clusters_phase_3", _RUNNER)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
clean_file = _mod.clean_file
default_output_path = _mod.default_output_path
validate_line = _mod.validate_line

VALID = "la{John Smith}\tla{JOHN SMITH}\tcy{Джон Смит}\tgg{X}"  # not all 12 tags, but structurally valid
VALID_FULL = "\t".join(f"{t}{{x}}" for t in ["la", "cy", "gk", "ab", "cn", "jp", "kr", "dv", "hb", "th", "gg", "am"])


def test_validate_accepts_full_cluster() -> None:
    ok, reason, tags = validate_line(VALID_FULL + "\n")
    assert ok
    assert reason == ""
    assert tags == _mod.TAGS


def test_validate_rejects_unterminated_cell() -> None:
    ok, reason, _ = validate_line("la{John Smith}\tcy{Джон")
    assert not ok
    assert reason == _mod.REASON_BAD_CELL


def test_validate_rejects_continuation_fragment() -> None:
    ok, reason, _ = validate_line("ك}\tcn{乔治·库利克}\tjp{ジョージ}")
    assert not ok
    assert reason == _mod.REASON_BAD_CELL


def test_validate_rejects_unknown_tag() -> None:
    ok, reason, _ = validate_line("la{John Smith}\tzz{x}")
    assert not ok
    assert reason == _mod.REASON_BAD_CELL


def test_validate_rejects_empty_value() -> None:
    ok, reason, _ = validate_line("la{John Smith}\tla{}\tcy{Джон}")
    assert not ok
    assert reason == _mod.REASON_EMPTY_VALUE


def test_validate_rejects_missing_la() -> None:
    ok, reason, _ = validate_line("cy{Джон Смит}\tgg{X}")
    assert not ok
    assert reason == _mod.REASON_NO_LA


def test_validate_rejects_blank_line() -> None:
    ok, reason, _ = validate_line("   \n")
    assert not ok
    assert reason == _mod.REASON_EMPTY


def test_clean_keeps_valid_drops_split_cluster(tmp_path: Path) -> None:
    # A cluster broken by an embedded newline: line 1 ends mid-cell, line 2
    # starts with the cell tail; both must be dropped.
    src = tmp_path / "in.txt"
    src.write_text(
        VALID_FULL
        + "\n"
        + "la{George J. Kulik}\tla{G. J. Kulik}\tla{Kulik, G. J.\n"
        + "ك}\tcn{乔治·库利克}\tjp{ジョージ・クーリック}\n"
        + VALID
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.txt"
    stats = clean_file(src, out)

    assert stats["total"] == 4
    assert stats["kept"] == 2
    assert stats["dropped"] == 2
    assert stats["by_reason"][_mod.REASON_BAD_CELL] == 2
    kept = out.read_text(encoding="utf-8").splitlines()
    assert kept == [VALID_FULL, VALID]
    assert stats["complete_clusters"] == 1
    assert stats["incomplete_clusters"] == 1


def test_clean_incomplete_kept_by_default_dropped_with_flag(tmp_path: Path) -> None:
    src = tmp_path / "in.txt"
    src.write_text(VALID + "\n", encoding="utf-8")

    out1 = tmp_path / "out1.txt"
    stats1 = clean_file(src, out1)
    assert stats1["kept"] == 1
    assert stats1["incomplete_clusters"] == 1

    out2 = tmp_path / "out2.txt"
    stats2 = clean_file(src, out2, drop_incomplete=True)
    assert stats2["kept"] == 0
    assert stats2["by_reason"][_mod.REASON_INCOMPLETE] == 1


def test_clean_gz_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "in.txt.gz"
    with gzip.GzipFile(src, mode="wb", mtime=0) as fh:
        fh.write((VALID_FULL + "\n" + "broken line\n").encode("utf-8"))
    out = tmp_path / "out.txt.gz"
    stats = clean_file(src, out)
    assert stats["kept"] == 1
    with gzip.open(out, "rt", encoding="utf-8") as fh:
        assert fh.read() == VALID_FULL + "\n"


def test_clean_empty_value_and_no_la_counted_separately(tmp_path: Path) -> None:
    src = tmp_path / "in.txt"
    src.write_text("la{}\tcy{x}\n" + "cy{Джон}\n", encoding="utf-8")
    out = tmp_path / "out.txt"
    stats = clean_file(src, out)
    assert stats["dropped"] == 2
    assert stats["by_reason"][_mod.REASON_EMPTY_VALUE] == 1
    assert stats["by_reason"][_mod.REASON_NO_LA] == 1


def test_default_output_path_inserts_clean_before_suffix() -> None:
    p = Path("/data/odin_train_set.txt.gz")
    assert default_output_path(p) == Path("/data/odin_train_set.clean.txt.gz")
    assert default_output_path(Path("/data/x.txt")) == Path("/data/x.clean.txt")
    assert default_output_path(Path("/data/x")) == Path("/data/x.clean")


@pytest.mark.parametrize("line", [VALID_FULL + "\n", VALID + "\n"])
def test_roundtrip_of_kept_lines_is_byte_identical(tmp_path: Path, line: str) -> None:
    src = tmp_path / "in.txt"
    src.write_text(line, encoding="utf-8")
    out = tmp_path / "out.txt"
    clean_file(src, out)
    assert out.read_bytes() == src.read_bytes()
