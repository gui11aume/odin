"""Tests for deterministic Latin company variants in runners/generate_companies_phase_1.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_RUNNER = Path(__file__).resolve().parents[1] / "runners" / "generate_companies_phase_1.py"
_spec = importlib.util.spec_from_file_location("generate_companies_phase_1", _RUNNER)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
company_latin_variants = _mod.company_latin_variants
title_org_name = _mod.title_org_name


def test_two_token_company_ltd_suffix() -> None:
    variants = company_latin_variants("Taiwan Semiconductor Manufacturing Company, Ltd.")
    assert "Taiwan Semiconductor Manufacturing Company, Ltd." in variants
    assert "Taiwan Semiconductor Manufacturing Co., Ltd." in variants
    assert "Taiwan Semiconductor Manufacturing Company Limited" in variants
    assert "Taiwan Semiconductor Manufacturing" in variants  # suffix dropped
    assert "TAIWAN SEMICONDUCTOR MANUFACTURING COMPANY, LTD." in variants  # ALL CAPS


def test_co_ltd_suffix_expands_to_company() -> None:
    variants = company_latin_variants("Samsung Electronics Co., Ltd.")
    assert "Samsung Electronics Co., Ltd." in variants
    assert "Samsung Electronics Company, Ltd." in variants
    assert "Samsung Electronics Ltd." in variants
    assert "Samsung Electronics" in variants
    assert "SAMSUNG ELECTRONICS CO., LTD." in variants


def test_ampersand_becomes_and() -> None:
    variants = company_latin_variants("Smith & Nephew, Inc.")
    assert "Smith & Nephew, Inc." in variants
    assert "Smith and Nephew, Inc." in variants
    assert "Smith & Nephew, Incorporated" in variants
    assert "SMITH & NEPHEW, INC." in variants
    assert "SMITH AND NEPHEW, INC." in variants


def test_ampersand_inside_token_is_untouched() -> None:
    variants = company_latin_variants("AT&T")
    assert "AT&T" in variants
    assert not any("and" in v for v in variants)


def test_hyphen_becomes_space() -> None:
    variants = company_latin_variants("Petro-Tex Chemical Corporation")
    assert "Petro-Tex Chemical Corporation" in variants
    assert "Petro Tex Chemical Corporation" in variants
    assert "PETRO-TEX CHEMICAL CORPORATION" in variants


def test_digit_tokens_keep_their_case() -> None:
    variants = company_latin_variants("3M")
    assert "3M" in variants
    assert "3m" not in variants


def test_diacritics_keep_strip_and_digraph() -> None:
    variants = company_latin_variants("Société Générale SA")
    assert "Société Générale SA" in variants
    assert "Societe Generale SA" in variants
    assert "Société Générale S.A." in variants
    assert "SOCIETE GENERALE SA" in variants


def test_llc_keeps_acronym_case_and_drops_suffix() -> None:
    variants = company_latin_variants("Bal Seal Engineering, LLC")
    assert "Bal Seal Engineering, LLC" in variants
    assert "Bal Seal Engineering, L.L.C." in variants
    assert "Bal Seal Engineering" in variants
    assert "BAL SEAL ENGINEERING, LLC" in variants
    assert not any("Llc" in v for v in variants)


def test_gmbh_and_ag_case() -> None:
    assert "GmbH" in title_org_name("siemens gmbh")
    assert "Siemens AG" in title_org_name("siemens ag")
    variants = company_latin_variants("Siemens AG")
    assert "Siemens AG" in variants
    assert "SIEMENS AG" in variants
    assert "Siemens" in variants


def test_no_suffix_is_never_invented() -> None:
    variants = company_latin_variants("Nike")
    assert variants == ["Nike", "NIKE"]
    assert not any(v.endswith((" Inc.", " LLC", " Ltd.")) for v in variants)


def test_empty_input() -> None:
    assert company_latin_variants("") == []
    assert company_latin_variants("   ") == []


def test_input_case_does_not_leak() -> None:
    variants = company_latin_variants("SAMSUNG ELECTRONICS CO., LTD.")
    assert "Samsung Electronics Co., Ltd." in variants
    assert "SAMSUNG ELECTRONICS CO., LTD." in variants
    assert not any(v.islower() for v in variants)


def test_dotted_accented_sarl_resolves_to_sarl() -> None:
    variants = company_latin_variants("Europe Brands S.à r.l.")
    assert "Europe Brands SARL" in variants
    assert "EUROPE BRANDS SARL" in variants
    assert "Europe Brands S.A.R.L." in variants
    assert "EUROPE BRANDS S.A.R.L." in variants
    assert "Europe Brands" in variants  # suffix dropped
    assert not any("S.À" in v or "S.A R.L" in v for v in variants)  # no mangled accent-dots forms


def test_dotted_suffixes_match_by_key() -> None:
    assert "Acme LLC" in company_latin_variants("Acme L.L.C.")
    assert "ACME LLC" in company_latin_variants("Acme L.L.C.")
    assert "Acme" in company_latin_variants("Acme L.L.C.")
    assert "Acme BV" in company_latin_variants("Acme B.V.")
    assert "Acme KK" in company_latin_variants("Acme K.K.")
    assert "Acme SARL" in company_latin_variants("Acme S.A.R.L.")
    assert "Acme OOO" in company_latin_variants("Acme O.O.O.")


def test_two_token_plain_suffixes() -> None:
    variants = company_latin_variants("Acme Company Limited")
    assert "Acme Co., Ltd." in variants
    assert "Acme Company, Limited" in variants
    assert "Acme" in variants
    variants = company_latin_variants("Acme Corporation, Ltd.")
    assert "Acme Corp., Ltd." in variants
    assert "Acme" in variants


def test_suffix_only_name_is_kept_whole() -> None:
    variants = company_latin_variants("LLC")
    assert variants == ["LLC"]


def test_variant_list_is_capped_and_deduped() -> None:
    variants = company_latin_variants("Taiwan Semiconductor Manufacturing Company, Ltd.")
    assert len(variants) == len(set(variants))
    assert len(variants) <= 64
