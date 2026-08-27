"""Tests for company cluster extraction in runners/extract_companies.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_RUNNER = Path(__file__).resolve().parents[1] / "runners" / "extract_companies.py"
_spec = importlib.util.spec_from_file_location("extract_companies", _RUNNER)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
canonical_key = _mod.canonical_key
group_companies = _mod.group_companies
build_line = _mod.build_line


def test_canonical_key_strips_case_punctuation_and_suffixes() -> None:
    assert canonical_key("IBM CORP.") == "ibm"
    assert canonical_key("Taiwan Semiconductor Manufacturing Company, Ltd.") == "taiwan semiconductor manufacturing"
    assert canonical_key("Smith & Nephew, Inc.") == "smith nephew"
    assert canonical_key("SAMSUNG Electronics Co., LTD") == "samsung electronics"


def test_canonical_key_suffix_only_is_empty() -> None:
    assert canonical_key("GmbH") == ""
    assert canonical_key("INC.") == ""
    assert canonical_key("") == ""


def test_canonical_key_keeps_non_suffix_tokens() -> None:
    assert canonical_key("a g a correa son l p") == "a g a correa son l p"
    assert canonical_key("3M") == "3m"


def test_group_companies_groups_and_orders() -> None:
    rows = [
        "Bal Seal Engineering, LLC",
        "BAL SEAL ENGINEERING, INC.",
        "Bal Seal Engineering, Inc.",
        "BAL SEAL ENGINEERING, LLC",
        "Bal Seal Engineering, LLC",
        "Sorel Corporation",
        "Sorel Corp.",
    ]
    groups = group_companies(rows)
    assert set(groups) == {"bal seal engineering", "sorel"}
    bal = groups["bal seal engineering"]
    # descending count, then descending length, then raw
    assert [raw for raw, _ in bal] == [
        "Bal Seal Engineering, LLC",
        "BAL SEAL ENGINEERING, INC.",
        "Bal Seal Engineering, Inc.",
        "BAL SEAL ENGINEERING, LLC",
    ]
    assert dict(groups["bal seal engineering"])["Bal Seal Engineering, LLC"] == 2
    assert dict(groups["sorel"]) == {"Sorel Corporation": 1, "Sorel Corp.": 1}
    # tie on count: longer raw first
    assert [raw for raw, _ in groups["sorel"]] == ["Sorel Corporation", "Sorel Corp."]


def test_group_companies_drops_empty_and_suffix_only() -> None:
    groups = group_companies(["", "   ", "Inc.", "GmbH", "Nike"])
    assert set(groups) == {"nike"}
    assert build_line([raw for raw, _ in groups["nike"]]) == "la{Nike}"


def test_build_line_format() -> None:
    line = build_line(["Nike", "NIKE", "Nike, Inc."])
    assert line == "la{Nike}\tla{NIKE}\tla{Nike, Inc.}"
