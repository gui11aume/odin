"""Tests for the inventor+company merge runner."""

from __future__ import annotations

import gzip
import importlib.util
from pathlib import Path

_RUNNER = Path(__file__).resolve().parents[1] / "runners" / "merge_odin_trainset.py"
_spec = importlib.util.spec_from_file_location("merge_odin_trainset", _RUNNER)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

merge = _mod.merge
drop_overlong_cells = _mod.drop_overlong_cells
pick_holdout = _mod.pick_holdout
format_cells = _mod.format_cells


class _FakeTokenizer:
    """One token per character: a cell's token count is 1 (tag) + len(value)."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [0] * len(text)

    def convert_tokens_to_ids(self, tok: str) -> int:
        return 1


def _read(path: Path) -> list[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return [ln.rstrip("\n") for ln in fh if ln.strip()]


def test_drop_overlong_cells_uses_tag_plus_surface() -> None:
    tok = _FakeTokenizer()
    cells = [("la", "abc"), ("la", "abcd"), ("cy", "абв")]
    kept, dropped = drop_overlong_cells(cells, tok, max_tokens=4)
    # 1 (tag token) + len(value): "abc" -> 4 kept; "abcd" -> 5 dropped; "абв" -> 4 kept
    assert kept == [("la", "abc"), ("cy", "абв")]
    assert dropped == 1
    kept, dropped = drop_overlong_cells([("la", "abcd")], tok, max_tokens=4)
    assert kept == [] and dropped == 1


def test_pick_holdout_deterministic_and_clamped() -> None:
    a = pick_holdout(100, 7, seed=123)
    b = pick_holdout(100, 7, seed=123)
    c = pick_holdout(100, 7, seed=124)
    assert a == b
    assert a != c
    assert len(a) == 7 and a.issubset(range(100))
    assert pick_holdout(10, 0, seed=1) == set()
    assert len(pick_holdout(5, 9, seed=1)) == 5


def test_format_cells_round_trip() -> None:
    cells = [("la", "A Corp"), ("cy", "А Корп"), ("la", "A CORP")]
    assert format_cells(cells) == "la{A Corp}\tcy{А Корп}\tla{A CORP}"


def test_merge_pool_val_and_pruning(tmp_path: Path) -> None:
    long60 = "x" * 60  # 1 + 60 > 48: pruned
    inventor = tmp_path / "inv.txt"
    inv_lines = [
        "la{A}\tla{AA}",  # i0: kept
        "la{B}\tla{BB}",  # i1: the old val line
        "la{C}\tla{CC}",  # i2: kept
        "la{D}\tla{" + long60 + "}",  # i3: long cell pruned, line kept as la{D}
    ]
    inventor.write_text("\n".join(inv_lines) + "\n", encoding="utf-8")

    companies = tmp_path / "comp.txt"
    comp_lines = [
        "la{E}",
        "la{F}\tla{" + "y" * 60 + "}",  # long cell pruned
        "la{G}",
        "la{H}",
    ]
    companies.write_text("\n".join(comp_lines) + "\n", encoding="utf-8")

    old_val = tmp_path / "old_val.txt"
    old_val.write_text("la{B}\tla{BB}\n", encoding="utf-8")

    pool_out = tmp_path / "pool.txt.gz"
    val_out = tmp_path / "val.txt.gz"
    tok = _FakeTokenizer()
    holdout = pick_holdout(4, 1, seed=123)
    stats = merge(
        inventor,
        companies,
        old_val,
        pool_out,
        val_out,
        holdout=1,
        seed=123,
        tokenizer=tok,
        max_tokens=48,
    )

    pool = _read(pool_out)
    val = _read(val_out)

    # Old val line never trained on; inventor order preserved; pruned cells gone.
    inventor_pool = ["la{A}\tla{AA}", "la{C}\tla{CC}", "la{D}"]
    # Company pool keeps file order, skips the holdout, prunes the long cell.
    comp_pool = []
    for i, ln in enumerate(comp_lines):
        if i in holdout:
            continue
        comp_pool.append(ln.replace("\tla{" + "y" * 60 + "}", "") if i == 1 else ln)
    assert pool == inventor_pool + comp_pool
    # Val: old val verbatim, then the single company holdout (in file order).
    assert val[0] == "la{B}\tla{BB}"
    assert len(val) == 2
    held = [comp_lines[i] for i in holdout][0]
    held_pruned = held.replace("\tla{" + "y" * 60 + "}", "") if holdout == {1} else held
    assert val[1] == held_pruned
    # The holdout line is not in the pool.
    assert held_pruned not in pool
    assert len(pool) == 3 + 3

    assert stats["inventor_skipped_val"] == 1
    assert stats["inventor_pruned_cells"] == 1
    assert stats["company_pruned_cells"] == 1
    assert stats["company_holdout"] == 1
    assert stats["pool_total"] == 6
    assert stats["val_total"] == 2
    assert stats["old_val"] == 1
