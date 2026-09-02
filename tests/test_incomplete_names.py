"""Reject incomplete extract outputs (initial-only / mononym / missing side)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_RUNNER = Path(__file__).resolve().parents[1] / "runners" / "generate_names_phase_1.py"
_spec = importlib.util.spec_from_file_location("generate_names_phase_1", _RUNNER)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_is_usable_person_name = _mod._is_usable_person_name
_dispatch_tool_call = _mod._dispatch_tool_call


def test_usable_requires_full_given_and_family() -> None:
    assert _is_usable_person_name("Guillaume", "Filion")
    assert _is_usable_person_name("Ernst A.", "Mayr")
    assert _is_usable_person_name("José María", "García-López")


def test_reject_initial_only_given() -> None:
    assert not _is_usable_person_name("G", "Filion")
    assert not _is_usable_person_name("G.", "Filion")
    assert not _is_usable_person_name("J.-M.", "Dupont")


def test_reject_mononym_and_empty_sides() -> None:
    assert not _is_usable_person_name("", "Madonna")
    assert not _is_usable_person_name("Madonna", "")
    assert not _is_usable_person_name("", "")
    assert not _is_usable_person_name("Filion", "")


def test_reject_initial_only_family() -> None:
    assert not _is_usable_person_name("Guillaume", "G")
    assert not _is_usable_person_name("Guillaume", "G.")


def test_dispatch_rejects_filion_g_as_incomplete() -> None:
    result = _dispatch_tool_call("extract_name", {"given": "G", "family": "Filion"})
    assert result == ("incomplete_name", None)


def test_dispatch_rejects_mononym_madonna() -> None:
    result = _dispatch_tool_call("extract_name", {"given": "", "family": "Madonna"})
    assert result == ("incomplete_name", None)


def test_dispatch_accepts_guillaume_filion() -> None:
    result = _dispatch_tool_call(
        "extract_name",
        {"given": "Guillaume", "family": "Filion"},
    )
    assert result is not None
    kind, variants = result
    assert kind == "extract_name"
    assert variants is not None
    assert "Guillaume Filion" in variants


def test_dispatch_accepts_patent_family_first_parses() -> None:
    """Gate must accept the splits the softened prompt teaches for USPTO lines."""
    cases = [
        ("James S.", "Cobb"),
        ("Thomas P.", "Guida"),
        ("Arthur O.", "Ernst"),
        ("Ron D.", "Wade"),
        ("Charles", "Loewe"),
        ("Daoxi", "Tan"),
        ("Dohiko", "Daniguchi"),
        ("Eileen M.", "Redmon"),
        ("Stephen B.", "Kong"),
    ]
    for given, family in cases:
        result = _dispatch_tool_call("extract_name", {"given": given, "family": family})
        assert result is not None, (given, family)
        kind, variants = result
        assert kind == "extract_name"
        assert variants


def test_prompt_teaches_patent_accept_and_narrow_reject() -> None:
    contract = _mod._EXTRACT_USER_CONTRACT
    assert "COBB JAMES S." in contract
    assert "KAUFMAN JAN" in contract
    assert "ALLISON CHARLOTTE C" in contract
    assert "NATOUR; GHALEB" in contract
    assert "J SPIELER KARL" in contract
    assert "semicolon" in contract
    assert "TAN DAOXI" in contract
    assert "When unsure between extract and reject, call extract_name" in contract
    assert "Do NOT reject ALL-CAPS" in contract
    assert "Titles / honorifics are NOT name parts" in contract
    assert "Peter Dr. Flury" in contract
    tools = _mod.TOOLS
    not_a = next(t for t in tools if t["function"]["name"] == "not_a_person_name")
    assert "COBB JAMES S." in not_a["function"]["description"]
    assert "KAUFMAN JAN" in not_a["function"]["description"]
