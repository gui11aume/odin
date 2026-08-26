"""Tests for deterministic Latin name variants in runners/generate_names_phase_1.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_RUNNER = Path(__file__).resolve().parents[1] / "runners" / "generate_names_phase_1.py"
_spec = importlib.util.spec_from_file_location("generate_names_phase_1", _RUNNER)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
latin_variants = _mod.latin_variants


def test_spaced_hyphen_given_does_not_crash() -> None:
    """LLM/OCR often emits 'Jean - Michel' with spaces around the hyphen."""
    variants = latin_variants("Jean - Michel", "Dupont")
    assert variants
    assert any("Jean Michel" in v or "Jean - Michel" in v for v in variants)


def test_bare_hyphen_middle_token_yields_first_plus_init() -> None:
    variants = latin_variants("Jean - Michel", "Dupont")
    assert "Jean M. Dupont" in variants


def test_normal_multi_given_still_works() -> None:
    variants = latin_variants("Ernst August", "Mayr")
    assert "Ernst A. Mayr" in variants
    assert "E.A. Mayr" in variants


def test_polish_l_stroke_folds_to_l() -> None:
    variants = latin_variants("Paweł", "Kowalski")
    assert "Paweł Kowalski" in variants
    assert "Pawel Kowalski" in variants


def test_nordic_ae_and_o_stroke() -> None:
    variants = latin_variants("Søren", "Kjær")
    assert "Søren Kjær" in variants
    assert "Soren Kjaer" in variants  # strip: ø→o, æ→ae
    assert "Soeren Kjaer" in variants  # digraph: ø→oe, then æ→ae


def test_icelandic_eth_and_thorn() -> None:
    variants = latin_variants("Þórður", "Jónsson")
    assert "Thordur Jonsson" in variants


def test_croatian_d_stroke() -> None:
    variants = latin_variants("Đorđe", "Đokić")
    assert "Dorde Dokic" in variants


def test_french_oe_ligature() -> None:
    variants = latin_variants("Jean", "Bœuf")
    assert "Jean Bœuf" in variants
    assert "Jean Boeuf" in variants


def test_turkish_dotless_i() -> None:
    variants = latin_variants("Işık", "Yıldız")
    assert "Isik Yildiz" in variants


def test_maltese_h_stroke() -> None:
    variants = latin_variants("Ħanna", "Borg")
    assert "Hanna Borg" in variants


def test_dutch_particle_both_casings() -> None:
    variants = latin_variants("Ludwig", "van Beethoven")
    assert "Ludwig Van Beethoven" in variants
    assert "Ludwig van Beethoven" in variants
    assert "Van Beethoven, Ludwig" in variants
    assert "van Beethoven, Ludwig" in variants


def test_french_particle_both_casings() -> None:
    variants = latin_variants("Charles", "de Gaulle")
    assert "Charles De Gaulle" in variants
    assert "Charles de Gaulle" in variants


def test_multi_particle_de_la() -> None:
    variants = latin_variants("Juan", "de la Cruz")
    assert "Juan De La Cruz" in variants
    assert "Juan de la Cruz" in variants


def test_no_particle_unchanged() -> None:
    variants = latin_variants("Ernst", "Mayr")
    assert "Ernst Mayr" in variants
    assert not any(" mayr" in v for v in variants)


def test_mc_surname_keeps_internal_capital() -> None:
    for family in ("MCMINN", "McMinn", "mcminn", "Mcminn"):
        variants = latin_variants("Wiley W", family)
        assert "Wiley W McMinn" in variants
        assert "WILEY W MCMINN" in variants
        assert not any("Mcminn" in v for v in variants)


def test_jr_in_given_is_not_initialed() -> None:
    variants = latin_variants("John Jr.", "Smith")
    assert "John Smith Jr." in variants
    assert "John Smith Jr" in variants
    assert "John Smith Junior" in variants
    assert "Smith, John Jr." in variants
    assert "J. J. Smith" not in variants
    assert "John J. Smith" not in variants
    assert "John Smith" not in variants


def test_jr_in_family_expands() -> None:
    variants = latin_variants("John", "Smith Jr")
    assert "John Smith Jr." in variants
    assert "John Smith Junior" in variants
    assert "J. Smith Jr." in variants
    assert "J. J. Smith" not in variants
