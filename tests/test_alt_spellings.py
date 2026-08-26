"""Tests for alt-Latin injection into latin_variants (phase 1 helpers / phase 2)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_RUNNER = Path(__file__).resolve().parents[1] / "runners" / "generate_names_phase_1.py"
_spec = importlib.util.spec_from_file_location("generate_names_phase_1", _RUNNER)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

MAX_ALT_LATIN = _mod.MAX_ALT_LATIN
_normalize_string_list = _mod._normalize_string_list
_split_display_name = _mod._split_display_name
latin_variants = _mod.latin_variants
latin_variants_with_alts = _mod.latin_variants_with_alts


def test_normalize_string_list_caps_and_dedupes() -> None:
    assert _normalize_string_list(["a", "a", "b", "c", "d"], 2) == ["a", "b"]
    assert _normalize_string_list('["x", "y"]', 3) == ["x", "y"]
    assert _normalize_string_list([], 3) == []
    assert _normalize_string_list("", 3) == []


def test_western_name_without_alts_unchanged() -> None:
    primary = latin_variants("Ernst A.", "Mayr")
    merged = latin_variants_with_alts("Ernst A.", "Mayr", [])
    assert merged == primary


def test_alt_latin_merges_yuri_for_yurij() -> None:
    variants = latin_variants_with_alts("Yurij P.", "Kirin", ["Yuri P. Kirin"])
    assert "Yurij P. Kirin" in variants
    assert "Yuri P. Kirin" in variants
    assert "YURI P. KIRIN" in variants
    assert "Kirin, Yuri P." in variants


def test_alt_latin_merges_chang_for_zhang() -> None:
    variants = latin_variants_with_alts("Wei", "Zhang", ["Wei Chang"])
    assert "Wei Zhang" in variants
    assert "Wei Chang" in variants
    assert "ZHANG, WEI" in variants or "Zhang, Wei" in variants
    assert "Chang, Wei" in variants


def test_alt_latin_cap_enforced() -> None:
    alts = [f"Alt{i} Name" for i in range(MAX_ALT_LATIN + 5)]
    capped = _normalize_string_list(alts, MAX_ALT_LATIN)
    assert len(capped) == MAX_ALT_LATIN
    variants = latin_variants_with_alts("Wei", "Zhang", alts)
    assert any("Alt0" in v for v in variants)
    assert not any(f"Alt{MAX_ALT_LATIN}" in v for v in variants)


def test_alt_latin_swapped_given_family_is_dropped() -> None:
    """'Yi Sedol' for primary Sedol Lee is given/family swap pollution."""
    variants = latin_variants_with_alts("Sedol", "Lee", ["Yi Sedol"])
    assert "Sedol Lee" in variants
    assert not any(v.startswith("Yi ") or ", Yi" in v for v in variants if "Sedol" in v)
    # The alt's family token is a primary given token → whole alt dropped.
    assert not any("Sedol" in v and "Yi" in v for v in variants)


def test_split_display_name_comma_and_western() -> None:
    assert _split_display_name("Yuri P. Kirin") == ("Yuri P.", "Kirin")
    assert _split_display_name("Kirin, Yuri P.") == ("Yuri P.", "Kirin")


def test_latin_variants_with_alts_base_merges_onto_existing_forms() -> None:
    """`base=` (phase 2's use) merges alts onto a pre-computed la{} list."""
    base = latin_variants("Viktor T.", "Skokov")
    merged = latin_variants_with_alts("Viktor T.", "Skokov", ["Victor T. Skokov"], base=base)
    assert merged[: len(base)] == base  # original forms untouched, in order
    assert "Victor T. Skokov" in merged
    assert "Skokov, Victor T." in merged


def test_latin_variants_with_alts_base_empty_alts_is_identity() -> None:
    base = latin_variants("Ernst A.", "Mayr")
    assert latin_variants_with_alts("Ernst A.", "Mayr", [], base=base) == base
