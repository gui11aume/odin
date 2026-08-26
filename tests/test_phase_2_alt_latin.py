"""Smoke tests for phase-2 alt-Latin parsing and cluster handling (no GPU / vLLM required)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_RUNNERS_DIR = Path(__file__).resolve().parents[1] / "runners"

_core_spec = importlib.util.spec_from_file_location("generate_names_core", _RUNNERS_DIR / "generate_names_core.py")
assert _core_spec is not None and _core_spec.loader is not None
core = importlib.util.module_from_spec(_core_spec)
_core_spec.loader.exec_module(core)

_p1_spec = importlib.util.spec_from_file_location("generate_names_phase_1", _RUNNERS_DIR / "generate_names_phase_1.py")
assert _p1_spec is not None and _p1_spec.loader is not None
p1 = importlib.util.module_from_spec(_p1_spec)
_p1_spec.loader.exec_module(p1)

_spec = importlib.util.spec_from_file_location("generate_names_phase_2", _RUNNERS_DIR / "generate_names_phase_2.py")
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_parse_emit_alt_latin = _mod._parse_emit_alt_latin
_split_cluster = _mod._split_cluster
_rebuild_cluster = _mod._rebuild_cluster

_SKOKOV_LINE = (
    "la{Viktor T. Skokov}\tla{VIKTOR T. SKOKOV}\tla{Skokov, Viktor T.}\tla{SKOKOV, VIKTOR T.}\t"
    "la{Viktor Skokov}\tla{VIKTOR SKOKOV}\tla{Skokov, Viktor}\tla{SKOKOV, VIKTOR}\t"
    "cy{Виктор Т. Скоков}\tgk{x}\tab{x}\tcn{x}\tjp{x}\tkr{x}\tdv{x}\thb{x}\tth{x}\tgg{x}\tam{x}"
)


def test_parse_emit_alt_latin_xml() -> None:
    raw = (
        "<tool_call>\n"
        "<function=emit_alt_latin>\n"
        '<parameter=alts>["Yuri P. Kirin", "Youri P. Kirin"]</parameter>\n'
        "</function>\n"
        "</tool_call>"
    )
    assert _parse_emit_alt_latin(raw) == ["Yuri P. Kirin", "Youri P. Kirin"]


def test_parse_emit_alt_latin_empty() -> None:
    raw = "<tool_call>\n<function=emit_alt_latin>\n<parameter=alts>[]</parameter>\n</function>\n</tool_call>"
    assert _parse_emit_alt_latin(raw) == []


def test_parse_emit_alt_latin_wrong_tool() -> None:
    raw = (
        "<tool_call>\n"
        "<function=extract_name>\n"
        "<parameter=given>Yuri</parameter>\n"
        "<parameter=family>Kirin</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    assert _parse_emit_alt_latin(raw) is None


def test_cluster_line_round_trips_through_phase1_formatter() -> None:
    """`parse_cluster_line` must be the exact inverse of phase 1's `_format_cluster`."""
    lat_vars = core.latin_variants("José María", "García-López")
    script_vars = {tag: [f"{tag}-form"] for tag in core.SCRIPTS}
    line = p1._format_cluster(lat_vars, script_vars)
    cells = core.parse_cluster_line(line)
    assert cells is not None
    assert core.format_cluster_line(cells) == line


def test_given_family_recovered_from_comma_form_handles_multitoken_family() -> None:
    lat_vars = core.latin_variants("Ivan", "Lopez Fernandez")
    given, family = core.given_family_from_la_forms(lat_vars)
    assert (given, family) == ("Ivan", "Lopez Fernandez")


def test_western_line_passthrough_is_byte_identical() -> None:
    cells = core.parse_cluster_line(_SKOKOV_LINE)
    assert cells is not None
    la_forms, other = _split_cluster(cells)
    given, family = core.given_family_from_la_forms(la_forms)
    merged = core.latin_variants_with_alts(given, family, [], base=la_forms)
    assert _rebuild_cluster(merged, other) == _SKOKOV_LINE


def test_non_western_line_gets_victor_alt_and_leaves_other_scripts_untouched() -> None:
    cells = core.parse_cluster_line(_SKOKOV_LINE)
    assert cells is not None
    la_forms, other = _split_cluster(cells)
    given, family = core.given_family_from_la_forms(la_forms)
    merged = core.latin_variants_with_alts(given, family, ["Victor T. Skokov"], base=la_forms)
    out_line = _rebuild_cluster(merged, other)

    assert la_forms == merged[: len(la_forms)]  # original la forms preserved, in order
    assert "Victor T. Skokov" in merged
    assert "Skokov, Victor T." in merged

    out_cells = core.parse_cluster_line(out_line)
    assert out_cells is not None
    non_la_out = [c for c in out_cells if c[0] != "la"]
    assert non_la_out == other  # non-Latin scripts untouched and in original order
