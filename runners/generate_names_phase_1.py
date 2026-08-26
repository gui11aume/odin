"""Generate cross-script name variant clusters for Odin training.

Pipeline:
  Name extraction -- Qwen3 tool call: `extract_name(given, family)`
                     or `not_a_person_name()` when the line has no person name
                     (greedy, thinking off; one retry on missing tool call)
  Latin variants  -- deterministic Python via the tool (Western word-order,
                     initials, Title/CAPS, diacritics, hyphen→space)
  Cross-script    -- Qwen3 tool call: `emit_scripts(...)` in 3 script-group
                     passes (A: cy/gk/hb/ab; B: cn/jp/kr; C: dv/th/gg/am);
                     optional jp_kana / *_alt as wired; incomplete →
                     per-missing-tag mini-retry, then drop
  Post-process    -- OpenCC Simplified↔Traditional for cn; CJK word-order
                     flips for short cn/kr names; cy/gk/hb order+initials;
                     drop few-shot contaminants

Input : gzipped or plain-text file, one candidate string per line (free-form;
        the model extracts given/family or rejects non-names).
Output: one tab-separated line per inventor in cluster-corpus format
        (complete clusters only — all 11 scripts required):
        la{Ernst A. Mayr}\tla{ERNST MAYR}\t...\tcy{Эрнст Майр}\tcn{...}
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any

from opencc import OpenCC

# Allow `python runners/generate_names_phase_1.py` to import the shared core.
_RUNNERS = Path(__file__).resolve().parent
if str(_RUNNERS) not in sys.path:
    sys.path.insert(0, str(_RUNNERS))

import generate_names_core as core  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)

# Re-exports used by tests / callers that still import this module.
SCRIPTS = core.SCRIPTS
MAX_NUM_SEQS = core.MAX_NUM_SEQS
MAX_EXTRACT_TOKENS = core.MAX_EXTRACT_TOKENS
MAX_ALT_LATIN = core.MAX_ALT_LATIN
MAX_MODEL_LEN = core.MAX_MODEL_LEN
TOOLS = core.EXTRACT_TOOLS
EXTRACT_SYSTEM_PROMPT = core.EXTRACT_SYSTEM_PROMPT
_EXTRACT_USER_CONTRACT = core._EXTRACT_USER_CONTRACT
latin_variants = core.latin_variants
latin_variants_with_alts = core.latin_variants_with_alts
SamplingParams = core.SamplingParams

# Private aliases so the rest of this file can keep a local `_foo` style.
_strip_diacritics = core.strip_diacritics
_title_name = core.title_name
_normalize_string_list = core.normalize_string_list
_split_display_name = core.split_display_name
_is_usable_person_name = core.is_usable_person_name
_parse_tool_calls = core.parse_tool_calls
_dispatch_tool_call = core.dispatch_extract_tool_call
_heal_given_family = core.heal_given_family
_build_llm = core.build_llm
_make_extract_sampling_params = core.make_extract_sampling_params
_make_extract_retry_sampling_params = core.make_extract_retry_sampling_params
_make_extract_user_message = core.make_extract_user_message
_make_extract_retry_user_message = core.make_extract_retry_user_message
_read_lines = core.read_lines
_display_name = core.display_name
_apply_chat = core.apply_chat
_prompt_fits = core.prompt_fits
_tokens_prompt = core.tokens_prompt
_strip_thinking = core.strip_thinking
_coerce_param_value = core.coerce_param_value
not_a_person_name = core.not_a_person_name

MAX_TRANSLITERATE_TOKENS = 512

_XLIT_PASSES: tuple[tuple[str, ...], ...] = (
    ("cy", "gk", "hb", "ab"),  # A: European + Arabic scripts
    ("cn", "jp", "kr"),  # B: CJK
    ("dv", "th", "gg", "am"),  # C: harder / lower-resource scripts
)
MAX_SCRIPT_ALT = 2
_SCRIPT_ALT_TAGS = ("cy", "ab", "cn", "jp", "kr")

# Few-shot CJK forms keyed by Latin labels they belong to. If these strings
# appear for a different input, treat as copy-paste contamination.
_FEWSHOT_FORMS: list[tuple[frozenset[str], frozenset[str]]] = [
    (
        frozenset({"guillaume j. filion", "guillaume filion"}),
        frozenset({"纪尧姆·菲利翁", "ギヨーム・フィリオン", "기욤 필리옹"}),
    ),
    (
        frozenset({"fatima al-hassan", "fatima al hassan"}),
        frozenset({"法蒂玛·哈桑", "ファティマ・アル＝ハッサン", "파티마 알-핫산"}),
    ),
    (
        frozenset({"josé maría garcía-lópez", "jose maria garcia-lopez"}),
        frozenset({"何塞·玛丽亚·加西亚-洛佩斯", "ホセ・マリア・ガルシア＝ロペス"}),
    ),
    (
        frozenset({"zhang wei", "wei zhang", "wei chang", "chang wei"}),
        frozenset({"张伟", "チョウ・ウェイ", "チャン・ウェイ", "장 웨이"}),
    ),
]
# Removed Yuki Tanaka few-shot, but models still copy these often.
_ORPHAN_CONTAMINANTS = frozenset({"田中雪", "タナカ ユキ", "다나카 유키"})

# ---------------------------------------------------------------------------
# Bibliographic order / initial variants (cy, gk, hb)
# ---------------------------------------------------------------------------

_HEBREW_GERESH = "\u05f3"  # ׳


def _order_and_initial_variants(given: str, family: str, inits: str) -> list[str]:
    """Family/given reorder plus initial abbreviations (up to five forms)."""
    if not family:
        return []
    candidates: list[str] = []
    if given:
        candidates.append(f"{family} {given}")
    if inits:
        candidates.append(f"{family} {inits}")
    if given:
        candidates.append(f"{given} {family}")
    if inits:
        candidates.append(f"{inits} {family}")
        candidates.append(f"{family}, {inits}")
    elif not given:
        candidates.append(family)
    return list(dict.fromkeys(c for c in candidates if c))


def _parse_two_part_name(text: str) -> tuple[str, str] | None:
    """Split a Western given–family string; last token is family.

    Multi-token givens (`Τόνι Γκόουαρντ Ζάργκερ`) keep everything but the
    last token as given.
    """
    tokens = text.split()
    if len(tokens) < 2:
        return None
    return " ".join(tokens[:-1]), tokens[-1]


_PATRONYMIC_SUFFIXES = (
    "ович",
    "евич",
    "овна",
    "евна",
    "ична",
    "инична",
    "івна",
)


def _is_patronymic(tok: str) -> bool:
    lower = tok.lower()
    return any(lower.endswith(sfx) for sfx in _PATRONYMIC_SUFFIXES)


def _cyrillic_initials(ymya: str, otchestvo: str) -> str:
    """Compact dotted initials: `И.`, `И.И.`, or `Т.Г.` for multi-token given."""
    parts: list[str] = []
    for tok in ymya.split():
        if tok:
            parts.append(f"{tok[0].upper()}.")
    if otchestvo:
        parts.append(f"{otchestvo[0].upper()}.")
    return "".join(parts)


def cyrillic_variants(ymya: str, otchestvo: str, familia: str) -> list[str]:
    """Return the most common Russian FIO spelling variants.

    Parameters
    ----------
    ymya:
        Given / first name (имя). May be multi-token for Western middle names.
    otchestvo:
        Patronymic (отчество). May be empty.
    familia:
        Family / surname (фамилия).

    Emits up to five forms commonly seen in Russian documents and
    bibliographies (Title Case only):

      1. Фамилия Имя Отчество   (or Фамилия Имя if no otchestvo)
      2. Фамилия И.О.           (or Фамилия И.)
      3. Имя Отчество Фамилия   (or Имя Фамилия)
      4. И.О. Фамилия           (or И. Фамилия)
      5. Фамилия, И.О.          (or Фамилия, И.)
    """
    ymya = _title_name(ymya.strip()) if ymya.strip() else ""
    otchestvo = _title_name(otchestvo.strip()) if otchestvo.strip() else ""
    familia = _title_name(familia.strip()) if familia.strip() else ""
    if not familia:
        return []

    given = f"{ymya} {otchestvo}".strip() if ymya else ""
    return _order_and_initial_variants(given, familia, _cyrillic_initials(ymya, otchestvo))


def _parse_cyrillic_fio(text: str) -> tuple[str, str, str] | None:
    """Split a Cyrillic name string into (ymya, otchestvo, familia).

    Handles 2-token given/family, 3-token FIO when a patronymic suffix is
    present, and Western given(+middle)+family (no patronymic) as
    (joined given, "", family). Returns None when unusable.
    """
    tokens = text.split()
    if len(tokens) == 2:
        return tokens[0], "", tokens[1]
    if len(tokens) == 3:
        if _is_patronymic(tokens[1]):
            return tokens[0], tokens[1], tokens[2]
        if _is_patronymic(tokens[2]):
            return tokens[1], tokens[2], tokens[0]
        # Western given + middle + family (middle is not an отчество).
        return f"{tokens[0]} {tokens[1]}", "", tokens[2]
    return None


def greek_variants(given: str, family: str) -> list[str]:
    """Return common Greek bibliographic spelling variants.

    Emits up to five Title Case forms:

      1. Family Given
      2. Family G.
      3. Given Family
      4. G. Family
      5. Family, G.
    """
    given = _title_name(given.strip()) if given.strip() else ""
    family = _title_name(family.strip()) if family.strip() else ""
    inits = f"{given[0].upper()}." if given else ""
    return _order_and_initial_variants(given, family, inits)


def _parse_greek_name(text: str) -> tuple[str, str] | None:
    """Split a Greek name into (given, family); last token is family."""
    return _parse_two_part_name(text)


def hebrew_variants(given: str, family: str) -> list[str]:
    """Return common Hebrew bibliographic spelling variants.

    Emits up to five forms (Hebrew has no case). Initials use geresh (׳):

      1. Family Given
      2. Family G׳
      3. Given Family
      4. G׳ Family
      5. Family, G׳
    """
    given = given.strip()
    family = family.strip()
    inits = f"{given[0]}{_HEBREW_GERESH}" if given else ""
    return _order_and_initial_variants(given, family, inits)


def _parse_hebrew_name(text: str) -> tuple[str, str] | None:
    """Split a Hebrew name into (given, family); last token is family."""
    return _parse_two_part_name(text)


# ---------------------------------------------------------------------------
# Unicode block validation (best-effort; warns only, never drops silently)
# ---------------------------------------------------------------------------

_SCRIPT_RANGES: dict[str, list[tuple[int, int]]] = {
    "cy": [(0x0400, 0x04FF), (0x0500, 0x052F)],
    "gk": [(0x0370, 0x03FF), (0x1F00, 0x1FFF)],
    "ab": [(0x0600, 0x06FF), (0x0750, 0x077F)],
    "cn": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],
    "jp": [(0x3040, 0x309F), (0x30A0, 0x30FF), (0x4E00, 0x9FFF)],
    "kr": [(0xAC00, 0xD7AF), (0x1100, 0x11FF)],
    "dv": [(0x0900, 0x097F)],
    "hb": [(0x0590, 0x05FF), (0xFB1D, 0xFB4F)],
    "th": [(0x0E00, 0x0E7F)],
    "gg": [(0x10A0, 0x10FF), (0x2D00, 0x2D2F)],
    "am": [(0x0530, 0x058F), (0xFB13, 0xFB17)],
}


def _has_script_chars(text: str, tag: str) -> bool:
    """True if every letter in `text` belongs to the target script.

    Punctuation, spaces, and digits are ignored. Rejects mixed forms like
    Thai+Latin (`ฟยอodor`) that an `any`-in-range check would wrongly accept.
    """
    ranges = _SCRIPT_RANGES.get(tag)
    if not ranges:
        return True
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    return all(any(lo <= ord(c) <= hi for lo, hi in ranges) for c in letters)


_SCRIPT_PARAM_DESCS: dict[str, str] = {
    "cy": "Phonetic Cyrillic (Russian-style). One given–family (or FIO) form only.",
    "gk": "Phonetic Greek. One given–family form only.",
    "ab": "Arabic-script rendering (Arabic/Persian/Urdu). One form only.",
    "cn": "Simplified Chinese phonetic (or conventional). One form only; Traditional is post-processed.",
    "jp": "Primary Japanese form: kanji for Japanese names, or katakana alone for foreign-only names.",
    "kr": "Hangul phonetic rendering. One form only.",
    "dv": "Devanagari phonetic rendering. One form only.",
    "hb": "Hebrew phonetic rendering. One given–family form only.",
    "th": "Thai phonetic rendering. One form only.",
    "gg": "Georgian phonetic rendering. One form only.",
    "am": "Armenian phonetic rendering. One form only.",
}

_SCRIPT_ALT_PARAM_DESCS: dict[str, str] = {
    "cy": (
        "Up to 2 common alternate Cyrillic spellings of the SAME person when the "
        "name is non-Russian/non-Slavic or Latin romanization is multi-standard. "
        "Empty for ordinary Russian/Slavic names. No order/initials variants."
    ),
    "ab": (
        "Up to 2 common alternate Arabic-script spellings of the SAME person when "
        "the name is foreign to Arabic script. Empty when unambiguous. No layout variants."
    ),
    "cn": (
        "Up to 2 common alternate Simplified Chinese phonetic spellings of the SAME "
        "person (different conventional characters), mainly for foreign names. Empty "
        "when one conventional form dominates. No Traditional or order flips."
    ),
    "jp": (
        "Up to 2 common alternate Japanese spellings of the SAME person (e.g. katakana "
        "forks for foreign names). Empty when unambiguous. Do not put kana here when "
        "jp_kana already covers the kana reading."
    ),
    "kr": (
        "Up to 2 common alternate Hangul spellings of the SAME person for foreign "
        "names with well-known forks. Empty for ordinary Korean names."
    ),
}


def _emit_scripts_tools(tags: tuple[str, ...] | list[str]) -> list[dict[str, Any]]:
    """Build emit_scripts tool requiring only `tags` (+ optional jp_kana / *_alt)."""
    properties: dict[str, Any] = {tag: {"type": "string", "description": _SCRIPT_PARAM_DESCS[tag]} for tag in tags}
    if "jp" in tags:
        properties["jp_kana"] = {
            "type": "string",
            "description": (
                "Optional second Japanese form (katakana or hiragana) "
                "when the name is Japanese or ambiguous. Empty string "
                "when jp already holds the sole foreign katakana form."
            ),
        }
    for tag in tags:
        if tag in _SCRIPT_ALT_PARAM_DESCS:
            properties[f"{tag}_alt"] = {
                "type": "array",
                "items": {"type": "string"},
                "description": _SCRIPT_ALT_PARAM_DESCS[tag],
            }
    return [
        {
            "type": "function",
            "function": {
                "name": "emit_scripts",
                "description": (
                    "Emit one primary phonetic (or conventional) transliteration per "
                    "script parameter for the Latin input name. Optionally add up to 2 "
                    "common alternate spellings via *_alt when wired. Do not invent "
                    "order/initials/case/Traditional variants — post-processing expands those."
                ),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(tags),
                },
            },
        },
    ]


def _xlit_system_prompt(tags: tuple[str, ...] | list[str]) -> str:
    """System prompt for an emit_scripts call covering only `tags`."""
    tag_list = ", ".join(tags)
    lines = [
        "You are a multilingual transliteration expert. Given an inventor name in Latin",
        f"script, call exactly one tool: emit_scripts(...) with parameters: {tag_list}.",
        "",
        "Scripts (one primary seed string each):",
    ]
    for tag in tags:
        lines.append(f"  {tag}  {_SCRIPT_NAMES[tag]}")
    if "jp" in tags:
        lines.extend(
            [
                "",
                "Optional jp_kana: katakana/hiragana when the name is Japanese or ambiguous;",
                "leave empty when jp already holds the sole foreign katakana form.",
            ]
        )
    alt_tags = [t for t in tags if t in _SCRIPT_ALT_PARAM_DESCS]
    if alt_tags:
        lines.extend(
            [
                "",
                f"Optional alt lists (up to 2 each): {', '.join(f'{t}_alt' for t in alt_tags)}.",
                "Use ONLY for well-known catalog/patent forks; prefer empty.",
            ]
        )
    lines.extend(
        [
            "",
            "Rules:",
            "1. Call emit_scripts exactly once. No free text.",
            "2. One primary form per required script parameter.",
            "3. Prefer the most common phonetic convention.",
            "4. CJK family-first when both parts are known.",
            "5. cn: Simplified Chinese only. Target script only — no Latin A-Z.",
        ]
    )
    return "\n".join(lines)


def _make_xlit_user_message(name: str, tags: tuple[str, ...] | list[str]) -> str:
    lines = ["Shape examples (argument values for Guillaume J. Filion):"]
    for tag in tags:
        lines.append(f"  {tag}={_SCRIPT_SHAPE_EXAMPLES[tag]}")
    lines.append("")
    lines.append(f"Transliterate: {name}")
    return "\n".join(lines)


_SCRIPT_NAMES: dict[str, str] = {
    "cy": "Cyrillic",
    "gk": "Greek",
    "ab": "Arabic",
    "cn": "Chinese",
    "jp": "Japanese",
    "kr": "Korean",
    "dv": "Devanagari",
    "hb": "Hebrew",
    "th": "Thai",
    "gg": "Georgian",
    "am": "Armenian",
}
# One few-shot-shaped example per script (script letters + punctuation only).
_SCRIPT_SHAPE_EXAMPLES: dict[str, str] = {
    "cy": "Гийом Ж. Филион",
    "gk": "Γκιγιόμ Ζ. Φιλιόν",
    "ab": "غيوم ج. فيليون",
    "cn": "纪尧姆·菲利翁",
    "jp": "ギヨーム・フィリオン",
    "kr": "기욤 필리옹",
    "dv": "गीयोम जे. फ़िलियों",
    "hb": "גיום ז׳. פיליון",
    "th": "กีโยม จ. ฟิลิออง",
    "gg": "გიომ ჟ. ფილიონი",
    "am": "Գիյոմ Ժ. Ֆիլիոն",
}


def _rejection_hint(tag: str, val: str) -> str:
    if any("A" <= c <= "Z" or "a" <= c <= "z" for c in val):
        return "contains Latin letters (A-Z)"
    return f"not valid {_SCRIPT_NAMES.get(tag, tag)} letters"


def _make_missing_scripts_message(
    name: str,
    missing: list[str],
    rejected: dict[str, list[str]] | None = None,
) -> str:
    lines = [f"Input: {name}"]
    for tag in missing:
        script = _SCRIPT_NAMES.get(tag, tag)
        bad = (rejected or {}).get(tag) or []
        example = _SCRIPT_SHAPE_EXAMPLES.get(tag, "")
        if bad:
            shown = ", ".join(repr(v) for v in bad)
            hints = "; ".join(_rejection_hint(tag, v) for v in bad)
            lines.append(
                f"Tag '{tag}' ({script}) was rejected ({shown}): {hints}. "
                f"Re-emit {script} letters only — no A-Z and no other scripts."
            )
        else:
            lines.append(f"Tag '{tag}' ({script}) was missing.")
        if example:
            lines.append(f"  Shape example for '{tag}': {example}")
    lines.append(
        "Call emit_scripts with exactly the missing tag parameter(s) listed above (do not emit other scripts)."
    )
    return "\n".join(lines)


def _make_xlit_sampling_params():
    # Structured emit_scripts tool call; low temperature, no thinking.
    return SamplingParams(
        max_tokens=MAX_TRANSLITERATE_TOKENS,
        temperature=0.2,
        top_p=0.95,
        stop=["</tool_call>"],
        include_stop_str_in_output=True,
    )


def _make_xlit_retry_sampling_params():
    # Escape repetitive wrong-script attractors on missing-tag retries.
    return SamplingParams(
        max_tokens=MAX_TRANSLITERATE_TOKENS,
        temperature=0.7,
        top_p=0.95,
        stop=["</tool_call>"],
        include_stop_str_in_output=True,
    )


def _dispatch_emit_scripts(args: dict[str, Any]) -> dict[str, list[str]]:
    """Map emit_scripts args to script_vars (primary seed + optional *_alt)."""
    out: dict[str, list[str]] = {}
    for tag in SCRIPTS:
        val = str(args.get(tag, "")).strip()
        if val:
            out[tag] = [val]
    for tag in _SCRIPT_ALT_TAGS:
        alts = _normalize_string_list(args.get(f"{tag}_alt", []), MAX_SCRIPT_ALT)
        if not alts:
            continue
        bucket = out.setdefault(tag, [])
        for alt in alts:
            if alt not in bucket:
                bucket.append(alt)
    jp_kana = str(args.get("jp_kana", "")).strip()
    if jp_kana:
        bucket = out.setdefault("jp", [])
        if jp_kana not in bucket:
            bucket.append(jp_kana)
    return out


def _parse_emit_scripts_output(raw: str) -> dict[str, list[str]] | None:
    """Parse an emit_scripts tool call; None if missing or wrong tool."""
    calls = _parse_tool_calls(raw)
    if not calls:
        return None
    name, args = calls[0]
    if name != "emit_scripts":
        return None
    return _dispatch_emit_scripts(args)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

_HAN_CHAR = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_HANGUL_CHAR = re.compile(r"[\uac00-\ud7af]+")
_CJK_SEP = re.compile(r"[\s·・＝=]+")


_S2T = OpenCC("s2t")
_T2S = OpenCC("t2s")


def _cn_script_forms(text: str) -> list[str]:
    """Simplified and Traditional forms via OpenCC."""
    forms = [text]
    for converted in (_S2T.convert(text), _T2S.convert(text)):
        if converted and converted not in forms:
            forms.append(converted)
    return forms


def _one_char_family_order_forms(text: str, tag: str) -> list[str]:
    """Family-first compact cn/kr names → also emit given-first flip.

    Assumes a single-character family when the string is 2–3 Han (cn) or
    Hangul (kr) characters with no separators. Length 4 is skipped so
    two-character surnames (e.g. 田川光一) are not mangled. Also swaps two
    tokens split by space / middot / katakana middle dot.
    """
    text = text.strip()
    forms = [text]

    bits = [b for b in _CJK_SEP.split(text) if b]
    if len(bits) == 2:
        sep_m = _CJK_SEP.search(text)
        sep = sep_m.group(0) if sep_m else " "
        flipped = f"{bits[1]}{sep}{bits[0]}"
        if flipped not in forms:
            forms.append(flipped)

    if tag == "cn" and _HAN_CHAR.fullmatch(text) and 2 <= len(text) <= 3:
        flipped = text[1:] + text[0]
        if flipped not in forms:
            forms.append(flipped)
    elif tag == "kr" and _HANGUL_CHAR.fullmatch(text) and 2 <= len(text) <= 3:
        flipped = text[1:] + text[0]
        if flipped not in forms:
            forms.append(flipped)

    return forms


def _cyrillic_yo_forms(text: str) -> list[str]:
    """Emit ё→е alternate (Фёдор → Федор). One-way only; е→ё is unsafe."""
    forms = [text]
    if "ё" in text or "Ё" in text:
        alt = text.replace("ё", "е").replace("Ё", "Е")
        if alt not in forms:
            forms.append(alt)
    return forms


def _expand_script_variants(script_vars: dict[str, list[str]]) -> dict[str, list[str]]:
    """Deterministic within-script expansions (OpenCC, CJK order, cy/gk/hb names)."""
    expanded: dict[str, list[str]] = {}
    for tag, values in script_vars.items():
        bucket: list[str] = []
        for val in values:
            seeds = [val]
            if tag == "cy":
                parsed = _parse_cyrillic_fio(val)
                if parsed:
                    seeds = list(dict.fromkeys([val, *cyrillic_variants(*parsed)]))
                seeds = [f for seed in seeds for f in _cyrillic_yo_forms(seed)]
            elif tag == "gk":
                parsed = _parse_greek_name(val)
                if parsed:
                    seeds = list(dict.fromkeys([val, *greek_variants(*parsed)]))
            elif tag == "hb":
                parsed = _parse_hebrew_name(val)
                if parsed:
                    seeds = list(dict.fromkeys([val, *hebrew_variants(*parsed)]))
            if tag in ("cn", "kr"):
                seeds = _one_char_family_order_forms(val, tag)
            if tag == "cn":
                seeds = [f for seed in seeds for f in _cn_script_forms(seed)]
            for form in seeds:
                if form not in bucket:
                    bucket.append(form)
        expanded[tag] = bucket
    return expanded


def _is_contaminant(label: str, val: str) -> bool:
    """True if `val` looks like a copied few-shot example for a different name."""
    label_key = _strip_diacritics(label).lower()
    for example_keys, forms in _FEWSHOT_FORMS:
        if label_key in example_keys:
            continue
        if val in forms or any(f in val for f in forms):
            return True
    if "tanaka" in label_key or "yuki" in label_key:
        return False
    return val in _ORPHAN_CONTAMINANTS or any(c in val for c in _ORPHAN_CONTAMINANTS)


def _filter_script_vars(
    label: str,
    script_vars: dict[str, list[str]],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Keep forms that match their script block; drop few-shot contaminants.

    Returns `(filtered, rejected)` where `rejected` maps tag → values that
    were present but dropped (wrong script or contaminant).
    """
    filtered: dict[str, list[str]] = {}
    rejected: dict[str, list[str]] = {}
    for tag in SCRIPTS:
        bucket: list[str] = []
        bad: list[str] = []
        for val in script_vars.get(tag, []):
            val = val.strip()
            if not val:
                continue
            if _is_contaminant(label, val):
                log.warning(f"Few-shot contaminant for tag '{tag}' on '{label}': '{val}'")
                if val not in bad:
                    bad.append(val)
                continue
            if not _has_script_chars(val, tag):
                log.warning(f"Script mismatch for tag '{tag}' on '{label}': '{val}'")
                if val not in bad:
                    bad.append(val)
                continue
            if val not in bucket:
                bucket.append(val)
        if bucket:
            filtered[tag] = bucket
        if bad:
            rejected[tag] = bad
    return filtered, rejected


def _missing_scripts(script_vars: dict[str, list[str]]) -> list[str]:
    return [t for t in SCRIPTS if not script_vars.get(t)]


def _merge_script_vars(
    base: dict[str, list[str]],
    extra: dict[str, list[str]],
) -> dict[str, list[str]]:
    merged = {t: list(vals) for t, vals in base.items()}
    for tag, vals in extra.items():
        bucket = merged.setdefault(tag, [])
        for val in vals:
            if val not in bucket:
                bucket.append(val)
    return merged


def _format_cluster(lat_vars: list[str], script_vars: dict[str, list[str]]) -> str:
    cells = [("la", v) for v in lat_vars]
    for tag in SCRIPTS:
        for val in script_vars.get(tag, []):
            cells.append((tag, val))
    return core.format_cluster_line(cells)


def _dispatch_extract_line(
    line: str,
    raw: str,
) -> tuple[str, str, str, list[str]] | None:
    """Parse one extraction completion into (line, given, family, lat_vars).

    Returns None when the line should be skipped (no/malformed tool call,
    non-name, or empty variants). Logs the reason.
    """
    calls = _parse_tool_calls(raw)
    if not calls:
        log.warning(f"No tool call for line {line!r}; skipping")
        return None
    tool_name, tool_args = calls[0]
    result = _dispatch_tool_call(tool_name, tool_args)
    if result is None:
        log.warning(f"Malformed tool call {tool_name!r} args={tool_args!r} for line {line!r}; skipping")
        return None
    kind, lat_vars = result
    if kind == "not_a_person_name":
        log.info(f"Rejected non-name line: {line!r}")
        return None
    if kind == "incomplete_name":
        log.info(f"Rejected incomplete name: {line!r}")
        return None
    assert lat_vars is not None
    if not lat_vars:
        log.warning(f"Empty variants for line {line!r}; skipping")
        return None
    given = str(tool_args.get("given", "")).strip()
    family = str(tool_args.get("family", "")).strip()
    return line, given, family, lat_vars


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate cross-script name variant clusters.")
    parser.add_argument(
        "input",
        help="Input text (.gz or plain): one candidate name string per line.",
    )
    parser.add_argument("--output", "-o", default="-", help="Output file (default: stdout).")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=MAX_NUM_SEQS,
        help="LLM batch size (default: %(default)s).",
    )
    args = parser.parse_args(argv)

    lines = list(_read_lines(args.input))
    log.info(f"Loaded {len(lines)} lines from {args.input}")

    llm = _build_llm()
    tokenizer = llm.get_tokenizer()
    extract_params = _make_extract_sampling_params()
    xlit_params = _make_xlit_sampling_params()
    xlit_retry_params = _make_xlit_retry_sampling_params()

    out_fh = open(args.output, "w", encoding="utf-8") if args.output != "-" else sys.stdout
    try:
        for batch_start in range(0, len(lines), args.batch_size):
            batch = lines[batch_start : batch_start + args.batch_size]

            # --- Phase 1: extract given/family via tool call ---
            extract_batch: list[tuple[str, list[int]]] = []
            for line in batch:
                prompt_ids = _apply_chat(
                    tokenizer,
                    [
                        {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
                        {"role": "user", "content": _make_extract_user_message(line)},
                    ],
                    tools=TOOLS,
                    enable_thinking=False,
                )
                if not _prompt_fits(prompt_ids, MAX_EXTRACT_TOKENS):
                    log.warning(f"Skipping overlong extract prompt ({len(prompt_ids)} tokens): {line!r:.80}")
                    continue
                extract_batch.append((line, prompt_ids))

            if not extract_batch:
                log.info(
                    f"Processed {min(batch_start + args.batch_size, len(lines))}/{len(lines)} (0 extract prompts fit)"
                )
                continue

            extract_outputs = llm.generate(
                [_tokens_prompt(ids) for _line, ids in extract_batch],
                extract_params,
                use_tqdm=False,
            )

            accepted: list[tuple[str, str, str, list[str]]] = []
            retry_lines: list[str] = []
            for (line, prompt_ids), output in zip(extract_batch, extract_outputs):
                raw = output.outputs[0].text
                if not _parse_tool_calls(raw):
                    retry_lines.append(line)
                    continue
                parsed = _dispatch_extract_line(line, raw)
                if parsed is not None:
                    accepted.append(parsed)

            # Retry with pointed feedback + elevated temperature.
            if retry_lines:
                extract_retry_params = _make_extract_retry_sampling_params()
                retry_batch: list[tuple[str, list[int]]] = []
                for line in retry_lines:
                    retry_ids = _apply_chat(
                        tokenizer,
                        [
                            {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
                            {"role": "user", "content": _make_extract_retry_user_message(line)},
                        ],
                        tools=TOOLS,
                        enable_thinking=False,
                    )
                    if _prompt_fits(retry_ids, MAX_EXTRACT_TOKENS):
                        retry_batch.append((line, retry_ids))
                retry_outputs = llm.generate(
                    [_tokens_prompt(ids) for _line, ids in retry_batch],
                    extract_retry_params,
                    use_tqdm=False,
                )
                for (line, _ids), output in zip(retry_batch, retry_outputs):
                    parsed = _dispatch_extract_line(line, output.outputs[0].text)
                    if parsed is not None:
                        accepted.append(parsed)

            if not accepted:
                log.info(f"Processed {min(batch_start + args.batch_size, len(lines))}/{len(lines)} (0 accepted)")
                continue

            # --- Phase 2: split-pass emit_scripts (A/B/C), then per-tag mini-retry ---
            ready: list[tuple[int, str, list[str]]] = []
            # (accepted_idx, label, lat_vars)
            for i, (_line, given, family, lat_vars) in enumerate(accepted):
                ready.append((i, _display_name(given, family), lat_vars))

            merged_by_idx: dict[int, dict[str, list[str]]] = {idx: {} for idx, _l, _lv in ready}

            for pass_tags in _XLIT_PASSES:
                tools = _emit_scripts_tools(pass_tags)
                sys_prompt = _xlit_system_prompt(pass_tags)
                pass_batch: list[tuple[int, str, list[int]]] = []
                for idx, label, _lat_vars in ready:
                    prompt_ids = _apply_chat(
                        tokenizer,
                        [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": _make_xlit_user_message(label, pass_tags)},
                        ],
                        tools=tools,
                        enable_thinking=False,
                    )
                    if not _prompt_fits(prompt_ids, MAX_TRANSLITERATE_TOKENS):
                        log.warning(
                            f"Skipping overlong transliterate prompt "
                            f"({len(prompt_ids)} tokens, tags={list(pass_tags)}): {label!r:.80}"
                        )
                        continue
                    pass_batch.append((idx, label, prompt_ids))
                if not pass_batch:
                    continue
                pass_outputs = llm.generate(
                    [_tokens_prompt(ids) for _i, _l, ids in pass_batch],
                    xlit_params,
                    use_tqdm=False,
                )
                for (idx, label, _ids), output in zip(pass_batch, pass_outputs):
                    parsed = _parse_emit_scripts_output(output.outputs[0].text)
                    if parsed is None:
                        log.warning(f"No emit_scripts call for '{label}' (tags={list(pass_tags)})")
                        continue
                    # Keep only this pass's tags (ignore any stray extras).
                    parsed = {t: v for t, v in parsed.items() if t in pass_tags}
                    merged_by_idx[idx] = _merge_script_vars(merged_by_idx[idx], parsed)

            pending: list[tuple[int, str, list[str], dict[str, list[str]], list[str], dict[str, list[str]]]] = []
            # (accepted_idx, label, lat_vars, script_vars, missing, rejected)
            written = 0
            for idx, label, lat_vars in ready:
                script_vars, rejected = _filter_script_vars(
                    label,
                    _expand_script_variants(merged_by_idx[idx]),
                )
                missing = _missing_scripts(script_vars)
                if missing:
                    log.warning(f"Missing scripts for '{label}': {missing}; mini-retry")
                    pending.append((idx, label, lat_vars, script_vars, missing, rejected))
                    continue
                print(_format_cluster(lat_vars, script_vars), file=out_fh)
                written += 1

            # Per-missing-tag mini-retry (narrow tool + retry sampling), then drop if incomplete.
            if pending:
                state: dict[
                    int,
                    tuple[str, list[str], dict[str, list[str]], dict[str, list[str]]],
                ] = {}
                # idx → (label, lat_vars, script_vars, rejected)
                mini_batch: list[tuple[int, str, str, list[int]]] = []
                # (accepted_idx, label, tag, prompt_ids)
                for idx, label, lat_vars, script_vars, missing, rejected in pending:
                    state[idx] = (label, lat_vars, script_vars, rejected)
                    for tag in missing:
                        tools = _emit_scripts_tools((tag,))
                        prompt_ids = _apply_chat(
                            tokenizer,
                            [
                                {"role": "system", "content": _xlit_system_prompt((tag,))},
                                {
                                    "role": "user",
                                    "content": _make_missing_scripts_message(label, [tag], rejected),
                                },
                            ],
                            tools=tools,
                            enable_thinking=False,
                        )
                        if not _prompt_fits(prompt_ids, MAX_TRANSLITERATE_TOKENS):
                            log.warning(
                                f"Skipping overlong mini-retry for '{label}' tag '{tag}' ({len(prompt_ids)} tokens)"
                            )
                            continue
                        mini_batch.append((idx, label, tag, prompt_ids))
                if mini_batch:
                    mini_outputs = llm.generate(
                        [_tokens_prompt(ids) for _i, _l, _t, ids in mini_batch],
                        xlit_retry_params,
                        use_tqdm=False,
                    )
                    for (idx, label, tag, _ids), output in zip(mini_batch, mini_outputs):
                        parsed = _parse_emit_scripts_output(output.outputs[0].text)
                        if parsed is None:
                            log.warning(f"No emit_scripts call on mini-retry for '{label}' tag '{tag}'")
                            continue
                        extra, extra_rej = _filter_script_vars(
                            label,
                            _expand_script_variants(parsed),
                        )
                        extra = {tag: extra[tag]} if tag in extra else {}
                        _lab, lat_vars, script_vars, rejected = state[idx]
                        script_vars = _merge_script_vars(script_vars, extra)
                        if tag in extra_rej:
                            rejected = {**rejected, tag: extra_rej[tag]}
                        state[idx] = (_lab, lat_vars, script_vars, rejected)
                for idx, (label, lat_vars, script_vars, _rejected) in state.items():
                    still_missing = _missing_scripts(script_vars)
                    if still_missing:
                        log.warning(
                            f"Dropping incomplete cluster for '{label}': still missing {still_missing} after mini-retry"
                        )
                        continue
                    print(_format_cluster(lat_vars, script_vars), file=out_fh)
                    written += 1

            log.info(
                f"Processed {min(batch_start + args.batch_size, len(lines))}/{len(lines)} "
                f"({len(accepted)} extracted, {written} written)"
            )
    finally:
        if out_fh is not sys.stdout:
            out_fh.close()


if __name__ == "__main__":
    main()
