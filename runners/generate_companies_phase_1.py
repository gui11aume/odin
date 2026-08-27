"""Generate company (organization) name variant clusters for Odin training.

Pipeline (same model and basic logic as generate_names_phase_1.py, adapted to
organizations; fewer axes because companies have far fewer surface variants
than person names):

  Latin variants   -- deterministic Python (Title/ALL-CAPS, legal-suffix
                      spellings, suffix drop, &/and, diacritics, hyphen->space),
                      merged with the observed raw forms
  Plausible scripts-- Qwen3 tool call: plausible_company_scripts(scripts);
                      which of the eleven non-Latin scripts this organization's
                      name could realistically appear in. Empty for ordinary
                      Western organizations (a person's name can appear in any
                      script; an organization's usually cannot).
  Cross-script     -- Qwen3 tool call: emit_scripts(...) for the chosen tags,
                      the same tool as generate_names_phase_1
  Post-process     -- OpenCC Simplified<->Traditional for cn; cy ё->е;
                      script-block validation (drop bad forms, never invent)

The goal is the exact way the assignee / applicant is written in a patent
document, including quirks of the jurisdiction or the clerk: case, legal-
suffix spelling, punctuation, OCR typos, trailing location/representation
clauses. The observed raw forms from extract_companies.py are the reference
surfaces; the generated variants augment them.

Input : gzipped or plain-text file of cluster lines from extract_companies.py
        (one organization per line, la{} cells only, ordered by frequency)
Output: one tab-separated line per organization in cluster-corpus format:
        la{...}\tla{...}\t...\tcn{...}\tjp{...}
        (non-Latin tags only when the LLM judged them plausible; a line with
        no plausible scripts is la{} only)

Usage:
    uv run python runners/generate_companies_phase_1.py /mnt/nvme1/odin_companies.raw.txt.gz \
        -o /mnt/nvme1/odin_companies.variants.txt.gz
    (--no-llm skips the LLM stages and writes la{}-only lines)
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any

# Allow `python runners/generate_companies_phase_1.py` to import the shared core.
_RUNNERS = Path(__file__).resolve().parent
if str(_RUNNERS) not in sys.path:
    sys.path.insert(0, str(_RUNNERS))

import generate_names_core as core  # noqa: E402
import generate_names_phase_1 as names_p1  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)

SCRIPTS = core.SCRIPTS
MAX_NUM_SEQS = core.MAX_NUM_SEQS
MAX_JUDGE_TOKENS = 128
MAX_TRANSLITERATE_TOKENS = names_p1.MAX_TRANSLITERATE_TOKENS

# Re-used from the person-name pipeline (script validation, OpenCC, ё->е,
# the emit_scripts tool builder, script tables).
_emit_scripts_tools = names_p1._emit_scripts_tools
_has_script_chars = names_p1._has_script_chars
_cn_script_forms = names_p1._cn_script_forms
_cyrillic_yo_forms = names_p1._cyrillic_yo_forms
_space_hyphens = core._space_hyphens
_parse_tool_calls = core.parse_tool_calls
_normalize_string_list = core.normalize_string_list
_apply_chat = core.apply_chat
_prompt_fits = core.prompt_fits
_tokens_prompt = core.tokens_prompt
read_lines = core.read_lines
SamplingParams = core.SamplingParams

# ---------------------------------------------------------------------------
# Deterministic Latin company variants
# ---------------------------------------------------------------------------

# Canonical display casing for legal / acronym tokens (Title Case alone would
# mangle LLC -> Llc, gmbh -> Gmbh, s.a. -> S.A.).
_ORG_TOKEN_CASE: dict[str, str] = {
    "llc": "LLC",
    "llp": "LLP",
    "plc": "PLC",
    "gmbh": "GmbH",
    "ag": "AG",
    "se": "SE",
    "oy": "Oy",
    "pte": "Pte.",
    "srl": "S.R.L.",
    "ooo": "OOO",
    "zoo": "zoo",
}


def _org_token_case(tok: str) -> str:
    bare = tok.strip(".,")
    if bare.lower() in _ORG_TOKEN_CASE:
        return _ORG_TOKEN_CASE[bare.lower()]
    if any(c.isdigit() or c == "&" for c in tok):
        return tok  # 3M / AT&T / H3C: keep as written
    return core.title_token(tok)


def title_org_name(text: str) -> str:
    """Title-case an organization name, keeping legal acronyms in canonical case."""
    return " ".join(_org_token_case(tok) for tok in text.split() if tok)


# Trailing legal suffixes, keyed by suffix key (lowercase, diacritics and
# punctuation stripped, glued: "S.à r.l." -> "sarl", "L.L.C." -> "llc",
# "Co., Ltd." -> "coltd"). Values are the display spellings of the suffix; the
# suffix-less form is always offered too, and the ALL-CAPS axis yields the
# glued uppercase form ("EUROPE BRANDS SARL").
_SUFFIX_FORMS: dict[str, tuple[str, ...]] = {
    "inc": ("Inc.", "Inc", "Incorporated"),
    "incorp": ("Incorp.", "Incorporated"),
    "incorporated": ("Inc.", "Inc", "Incorporated"),
    "ltd": ("Ltd.", "Ltd", "Limited"),
    "limited": ("Ltd.", "Ltd", "Limited"),
    "llc": ("LLC", "L.L.C.", "L.L.C"),
    "llp": ("LLP", "L.L.P."),
    "lp": ("L.P.", "L.P"),
    "corp": ("Corp.", "Corp", "Corporation"),
    "corporation": ("Corp.", "Corp", "Corporation"),
    "co": ("Co.", "Co", "Company"),
    "company": ("Co.", "Co", "Company"),
    "pte": ("Pte.", "Pte"),
    "plc": ("PLC", "P.L.C."),
    "sa": ("SA", "S.A."),
    "nv": ("NV", "N.V."),
    "bv": ("BV", "B.V."),
    "kk": ("KK", "K.K."),
    "kaisha": ("Kaisha", "K.K."),
    "ooo": ("OOO", "O.O.O."),
    "ab": ("AB", "Ab"),
    "spz": ("Sp.",),
    "gmbh": ("GmbH",),
    "ag": ("AG",),
    "se": ("SE",),
    "oy": ("Oy", "OY"),
    "zoo": ("zoo", "ZOO"),
    "aktiengesellschaft": ("Aktiengesellschaft", "AG", "A.G."),
    "sarl": ("SARL", "S.A.R.L.", "Société à responsabilité limitée"),
    "companyltd": ("Company, Ltd.", "Co., Ltd.", "Co Ltd.", "Company Limited", "Ltd."),
    "coltd": ("Co., Ltd.", "Co Ltd.", "Co., Limited", "Company, Ltd.", "Ltd."),
    "companylimited": ("Company Limited", "Company, Limited", "Co., Ltd.", "Ltd."),
    "colimited": ("Co., Ltd.", "Co Ltd.", "Co., Limited", "Ltd."),
    "corpltd": ("Corp., Ltd.", "Corporation, Ltd.", "Ltd."),
    "corporationltd": ("Corp., Ltd.", "Corporation, Ltd.", "Ltd."),
    "pteltd": ("Pte. Ltd.", "Pte Ltd.", "Pty. Ltd."),
    "kabushikikaisha": ("Kabushiki Kaisha", "K.K.", "KK"),
}


def _suffix_key(*tokens: str) -> str:
    """Suffix key of trailing tokens: lowercase, diacritics and punctuation
    stripped, glued ("S.à r.l." -> "sarl", "L.L.C." -> "llc")."""
    joined = core.strip_diacritics(" ".join(tokens).lower())
    return re.sub(r"[^a-z0-9]+", "", joined)


def _peel_suffix(tokens: list[str]) -> tuple[list[str], tuple[str, ...]]:
    """Peel a trailing one- or two-token legal suffix.

    Matching is on the suffix key, so dotted and accented surfaces
    ("S.à r.l.", "L.L.C.", "B.V.", "S.A.R.L.") all resolve to the same entry.

    Returns (core tokens, suffix surfaces). The suffix surfaces exclude the
    dropped form; the caller adds the core alone. Returns the input tokens
    unchanged with an empty surface tuple when there is no suffix to peel.
    """
    if len(tokens) >= 3:
        key = _suffix_key(tokens[-2], tokens[-1])
        if key in _SUFFIX_FORMS:
            return tokens[:-2], _SUFFIX_FORMS[key]
    if len(tokens) >= 2:
        key = _suffix_key(tokens[-1])
        if key in _SUFFIX_FORMS:
            return tokens[:-1], _SUFFIX_FORMS[key]
    return tokens, ()


def company_latin_variants(display: str) -> list[str]:
    """Return plausible Latin-script surface variants of an organization name.

    `display` is the organization name in any case; it is title-cased first so
    ALL-CAPS / mixed-case inputs do not leak into the output set.

    Axes (only what patent clerks actually vary for organizations; there is no
    word order, no initials, no particle axis):

      case          -- Title Case | ALL CAPS (no all-lowercase)
      legal suffix  -- the observed spelling | alternate spellings | dropped
                      (Co./Co/Company, Ltd./Ltd/Limited, Inc./Incorporated,
                      LLC/L.L.C., S.à r.l./S.A.R.L./SARL, ...; two-token units
                      like "Company, Ltd."). Suffixes are matched on a
                      punctuation- and diacritic-insensitive key, so dotted and
                      accented surfaces resolve to the same entry.
      ampersand     -- `&` | `and` (standalone token only; AT&T is untouched)
      diacritics    -- keep | strip | German/Nordic digraph
      compound sep  -- hyphen | space

    Suffixes are never invented: a name without a legal suffix never gains one.
    """
    display = " ".join(display.split())
    if not display:
        return []

    tokens = display.split()
    core_toks, suffixes = _peel_suffix(tokens)
    if not core_toks:
        core_toks, suffixes = tokens, ()  # suffix-only name: keep it whole

    core_name = " ".join(_org_token_case(t) for t in core_toks)

    base: list[str] = []
    for sfx in suffixes:
        base.append(f"{core_name} {sfx}")
    # Dropped-suffix (or whole, when nothing was peeled). The separator comma
    # before the suffix belongs to the joined forms, not to the standalone one.
    base.append(core_name.rstrip(".,") or core_name)

    # Diacritics: keep, strip, German/Nordic digraph.
    diac: list[str] = []
    for bf in base:
        diac.append(bf)
        s = core.strip_diacritics(bf)
        if s != bf:
            diac.append(s)
        d = core.digraph(bf)
        if d != bf and d != s:
            diac.append(d)

    # Compound separators: hyphen -> space (initialisms like L.L.C. stay).
    sep: list[str] = []
    for v in diac:
        sep.append(v)
        spaced = _space_hyphens(v)
        if spaced != v:
            sep.append(spaced)

    # Ampersand: standalone `&` -> `and`.
    amp: list[str] = []
    for v in sep:
        amp.append(v)
        if re.search(r"\s&\s", v):
            and_form = re.sub(r"\s&\s", " and ", v)
            if and_form not in amp:
                amp.append(and_form)

    # Case: Title Case (already) + ALL CAPS. Skip all-lowercase.
    cased: list[str] = []
    for v in amp:
        cased.append(v)
        u = v.upper()
        if u != v:
            cased.append(u)

    seen: set[str] = set()
    out: list[str] = []
    for v in cased:
        v = " ".join(v.split())
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out[:64]


# ---------------------------------------------------------------------------
# Plausible-scripts judge (LLM)
# ---------------------------------------------------------------------------

PLAUSIBLE_SCRIPTS_TOOL: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "plausible_company_scripts",
            "description": (
                "Call once with the subset (possibly empty) of the eleven non-Latin "
                "scripts in which this organization's name could realistically appear "
                "in a patent document."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "scripts": {
                        "type": "array",
                        "items": {"type": "string", "enum": SCRIPTS},
                        "description": (
                            "Subset (possibly empty) of: "
                            + ", ".join(SCRIPTS)
                            + ". Empty for ordinary Western organizations."
                        ),
                    }
                },
                "required": ["scripts"],
            },
        },
    },
]

JUDGE_SYSTEM_PROMPT = (
    "You are a patent-document expert. The input is an organization name as written in "
    "Latin script. Call exactly one tool: plausible_company_scripts(scripts=[...]). No free text."
)

_JUDGE_CONTRACT = """\
Decide in which of the eleven non-Latin scripts this organization's name could \
realistically appear in a patent document. Two reasons qualify:
  1. The organization is from a jurisdiction that writes in that script \
(its name appears in that script in local documents); OR
  2. The organization has a well-established conventional name in that \
script's language, used in that jurisdiction's business/patent records.

Western organizations qualify under (2) when they are internationally famous \
and have a fixed local name (Apple -> 苹果/アップル/애플, Google -> 谷歌/グーグル/구글, \
Siemens -> 西门子/シーメンス/시멘스, General Electric -> 通用电气/ゼネラル・エレクトリック). \
Ordinary, non-famous Western organizations do NOT: in Chinese/Japanese/Korean \
documents they are usually left in Latin script, so they get an empty list.

Do not pad: include a script only when (1) or (2) genuinely holds. Most \
organizations get an empty or short list.

Scripts: cy Cyrillic, gk Greek, ab Arabic, cn Chinese, jp Japanese, kr Korean, \
dv Devanagari, hb Hebrew, th Thai, gg Georgian, am Armenian.

Examples:
"Samsung Electronics Co., Ltd." -> ["cn", "jp", "kr"]
"Honda Motor Co., Ltd." -> ["cn", "jp", "kr"]
"Siemens AG" -> ["cy", "cn", "jp", "kr"]
"Apple Inc." -> ["cn", "jp", "kr"]
"General Electric" -> ["cn", "jp", "kr"]
"Saudi Arabian Oil Company" -> ["ab"]
"Bal Seal Engineering, LLC" -> []
"Acme Fasteners LLC" -> []

### Input
"""


def make_judge_user_message(display: str) -> str:
    return f"{_JUDGE_CONTRACT}{display}"


def make_judge_retry_user_message(display: str) -> str:
    return f"{_JUDGE_CONTRACT}{display}\n\nYou MUST respond with a tool call. No free text."


def _make_judge_sampling_params():
    return SamplingParams(
        max_tokens=MAX_JUDGE_TOKENS,
        temperature=0.0,
        top_p=1.0,
        stop=["</tool_call>"],
        include_stop_str_in_output=True,
    )


def _make_judge_retry_sampling_params():
    return SamplingParams(
        max_tokens=MAX_JUDGE_TOKENS,
        temperature=0.3,
        top_p=0.95,
        stop=["</tool_call>"],
        include_stop_str_in_output=True,
    )


def parse_plausible_scripts(raw: str) -> list[str] | None:
    """Parse the judge call; valid tag list, or None when no usable tool call."""
    calls = _parse_tool_calls(raw)
    if not calls:
        return None
    name, args = calls[0]
    if name != "plausible_company_scripts":
        return None
    valid = set(SCRIPTS)
    seen: set[str] = set()
    out: list[str] = []
    for tag in _normalize_string_list(args.get("scripts"), len(SCRIPTS)):
        if tag in valid and tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


# ---------------------------------------------------------------------------
# Cross-script emission (LLM, same emit_scripts tool as the person-name phase)
# ---------------------------------------------------------------------------


def _xlit_system_prompt(tags: tuple[str, ...]) -> str:
    tag_list = ", ".join(tags)
    lines = [
        "You are a patent-document expert. Given an organization (company / assignee) name in",
        f"Latin script, call exactly one tool: emit_scripts(...) with parameters: {tag_list}.",
        "",
        "Scripts (one form each):",
    ]
    for tag in tags:
        lines.append(f"  {tag}  {names_p1._SCRIPT_NAMES[tag]}")
    lines.extend(
        [
            "",
            "Rules:",
            "1. Call emit_scripts exactly once. No free text.",
            "2. One form per required script parameter.",
            "3. Emit the form in which this organization's name appears in patent documents",
            "   from that jurisdiction.",
            "4. Transliterate the organization name only. Ignore trailing location,",
            '   representation, or affiliation clauses (e.g. "AT BEN-GURION UNIVERSITY",',
            '   "AS REPRESENTED BY THE SECRETARY OF ...").',
            "5. Use the established conventional form when one clearly exists (e.g. Honda Motor -> ホンダ).",
            "6. Keep the local conventional legal suffix when the jurisdiction uses one",
            "   (cy: ПАО/ООО, kr: 주식회사, cn: 有限公司); drop Latin suffixes otherwise.",
            "7. cn: Simplified Chinese only. Target script only - no Latin A-Z.",
        ]
    )
    return "\n".join(lines)


def _make_xlit_user_message(display: str, tags: tuple[str, ...]) -> str:
    lines = ["Shape examples (script shape only; the values show a person name):"]
    for tag in tags:
        lines.append(f"  {tag}={names_p1._SCRIPT_SHAPE_EXAMPLES[tag]}")
    lines.extend(["", f"Transliterate organization: {display}"])
    return "\n".join(lines)


def _make_xlit_sampling_params():
    return SamplingParams(
        max_tokens=MAX_TRANSLITERATE_TOKENS,
        temperature=0.2,
        top_p=0.95,
        stop=["</tool_call>"],
        include_stop_str_in_output=True,
    )


def _parse_emit_scripts_output(raw: str) -> dict[str, list[str]] | None:
    return names_p1._parse_emit_scripts_output(raw)


def _expand_company_script_variants(script_vars: dict[str, list[str]]) -> dict[str, list[str]]:
    """Deterministic within-script expansion for organization forms.

    OpenCC Simplified<->Traditional for cn and cy ё->е only. The person-name
    order/initials/FIO expanders do not apply to organizations.
    """
    expanded: dict[str, list[str]] = {}
    for tag, values in script_vars.items():
        bucket: list[str] = []
        for val in values:
            seeds = [val]
            if tag == "cy":
                seeds = [f for s in seeds for f in _cyrillic_yo_forms(s)]
            if tag == "cn":
                seeds = [f for s in seeds for f in _cn_script_forms(s)]
            for form in seeds:
                if form not in bucket:
                    bucket.append(form)
        expanded[tag] = bucket
    return expanded


def _format_cluster(lat_vars: list[str], script_vars: dict[str, list[str]]) -> str:
    cells = [("la", v) for v in lat_vars]
    for tag in SCRIPTS:
        for val in script_vars.get(tag, []):
            cells.append((tag, val))
    return core.format_cluster_line(cells)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _process_line_deterministic(line: str) -> tuple[str, list[str]] | None:
    """Parse one input cluster line; return (display, la cells).

    Returns None when the line is not a usable company cluster (not cluster
    format, has a non-la cell, or has no cells at all).
    """
    parsed = core.parse_cluster_line(line)
    if parsed is None:
        log.warning(f"Not cluster format, skipping: {line!r:.80}")
        return None
    if any(tag != "la" for tag, _val in parsed):
        log.warning(f"Non-la cell in company line, passing through: {line!r:.80}")
        return None
    observed = [val for _tag, val in parsed]
    if not observed:
        return None
    display = title_org_name(observed[0])
    gen = company_latin_variants(display)
    lat_vars = list(dict.fromkeys([*observed, *gen]))
    return display, lat_vars


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate company name variant clusters.")
    parser.add_argument("input", help="Input text (.gz or plain): company cluster lines from extract_companies.py.")
    parser.add_argument("--output", "-o", default="-", help="Output file (default: stdout).")
    parser.add_argument("--batch-size", type=int, default=MAX_NUM_SEQS, help="LLM batch size (default: %(default)s).")
    parser.add_argument("--no-llm", action="store_true", help="Skip the LLM stages; write la{}-only lines.")
    args = parser.parse_args(argv)

    lines = list(read_lines(args.input))
    log.info(f"Loaded {len(lines)} company lines from {args.input}")

    if not args.no_llm:
        llm = core.build_llm()
        tokenizer = llm.get_tokenizer()
        judge_params = _make_judge_sampling_params()
        judge_retry_params = _make_judge_retry_sampling_params()
        xlit_params = _make_xlit_sampling_params()
    else:
        llm = tokenizer = judge_params = judge_retry_params = xlit_params = None

    out_fh = open(args.output, "w", encoding="utf-8") if args.output != "-" else sys.stdout
    try:
        for batch_start in range(0, len(lines), args.batch_size):
            batch = lines[batch_start : batch_start + args.batch_size]

            ready: list[tuple[str, list[str]]] = []  # (display, lat_vars)
            passthrough: list[str] = []
            for line in batch:
                parsed = _process_line_deterministic(line)
                if parsed is None:
                    if core.parse_cluster_line(line) is not None:
                        passthrough.append(line)
                    continue
                ready.append(parsed)

            judge_by_display: dict[str, list[str]] = {}
            if ready and not args.no_llm:
                assert llm is not None and tokenizer is not None
                assert judge_params is not None and judge_retry_params is not None and xlit_params is not None
                # --- Plausible-scripts judge ---
                judge_batch: list[tuple[str, list[int]]] = []
                for display, _lat in ready:
                    prompt_ids = _apply_chat(
                        tokenizer,
                        [
                            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                            {"role": "user", "content": make_judge_user_message(display)},
                        ],
                        tools=PLAUSIBLE_SCRIPTS_TOOL,
                        enable_thinking=False,
                    )
                    if not _prompt_fits(prompt_ids, MAX_JUDGE_TOKENS):
                        log.warning(f"Skipping overlong judge prompt ({len(prompt_ids)} tokens): {display!r:.80}")
                        continue
                    judge_batch.append((display, prompt_ids))

                judge_outputs = llm.generate(
                    [_tokens_prompt(ids) for _d, ids in judge_batch], judge_params, use_tqdm=False
                )
                retry_displays: list[str] = []
                for (display, _ids), output in zip(judge_batch, judge_outputs):
                    scripts = parse_plausible_scripts(output.outputs[0].text)
                    if scripts is None:
                        retry_displays.append(display)
                        continue
                    judge_by_display[display] = scripts

                if retry_displays:
                    retry_batch: list[tuple[str, list[int]]] = []
                    for display in retry_displays:
                        prompt_ids = _apply_chat(
                            tokenizer,
                            [
                                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                                {"role": "user", "content": make_judge_retry_user_message(display)},
                            ],
                            tools=PLAUSIBLE_SCRIPTS_TOOL,
                            enable_thinking=False,
                        )
                        if _prompt_fits(prompt_ids, MAX_JUDGE_TOKENS):
                            retry_batch.append((display, prompt_ids))
                    retry_outputs = llm.generate(
                        [_tokens_prompt(ids) for _d, ids in retry_batch], judge_retry_params, use_tqdm=False
                    )
                    for (display, _ids), output in zip(retry_batch, retry_outputs):
                        scripts = parse_plausible_scripts(output.outputs[0].text)
                        if scripts is not None:
                            judge_by_display[display] = scripts

                # --- Cross-script emission for the judged tags ---
                script_vars_by_display: dict[str, dict[str, list[str]]] = {}
                emit_groups: dict[tuple[str, ...], list[tuple[str, list[int]]]] = {}
                for display, _lat in ready:
                    tags = tuple(t for t in SCRIPTS if t in judge_by_display.get(display, ()))
                    if not tags:
                        continue
                    prompt_ids = _apply_chat(
                        tokenizer,
                        [
                            {"role": "system", "content": _xlit_system_prompt(tags)},
                            {"role": "user", "content": _make_xlit_user_message(display, tags)},
                        ],
                        tools=_emit_scripts_tools(tags),
                        enable_thinking=False,
                    )
                    if not _prompt_fits(prompt_ids, MAX_TRANSLITERATE_TOKENS):
                        log.warning(
                            f"Skipping overlong transliterate prompt "
                            f"({len(prompt_ids)} tokens, tags={list(tags)}): {display!r:.80}"
                        )
                        continue
                    emit_groups.setdefault(tags, []).append((display, prompt_ids))

                for tags, group in emit_groups.items():
                    group_outputs = llm.generate(
                        [_tokens_prompt(ids) for _d, ids in group], xlit_params, use_tqdm=False
                    )
                    for (display, _ids), output in zip(group, group_outputs):
                        parsed = _parse_emit_scripts_output(output.outputs[0].text)
                        if parsed is None:
                            log.warning(f"No emit_scripts call for '{display}' (tags={list(tags)})")
                            continue
                        parsed = {t: v for t, v in parsed.items() if t in tags}
                        script_vars_by_display.setdefault(display, {}).update(parsed)

                # --- Filter; mini-retry judged tags that came back empty or were rejected ---
                pending: list[tuple[str, tuple[str, ...], dict[str, list[str]]]] = []
                # (display, tags, rejected values per tag)
                for display, raw_vars in script_vars_by_display.items():
                    filtered, rejected = names_p1._filter_script_vars(
                        display, _expand_company_script_variants(raw_vars)
                    )
                    script_vars_by_display[display] = filtered
                    wanted = judge_by_display.get(display, ())
                    missing = [t for t in SCRIPTS if t in wanted and not filtered.get(t)]
                    if missing:
                        pending.append(
                            (
                                display,
                                tuple(t for t in SCRIPTS if t in missing),
                                {t: rejected.get(t, []) for t in missing},
                            )
                        )

                if pending:
                    mini_params = names_p1._make_xlit_retry_sampling_params()
                    mini_batch: list[tuple[str, tuple[str, ...], list[int]]] = []
                    for display, tags, rejected in pending:
                        prompt_ids = _apply_chat(
                            tokenizer,
                            [
                                {"role": "system", "content": _xlit_system_prompt(tags)},
                                {
                                    "role": "user",
                                    "content": names_p1._make_missing_scripts_message(display, list(tags), rejected),
                                },
                            ],
                            tools=_emit_scripts_tools(tags),
                            enable_thinking=False,
                        )
                        if not _prompt_fits(prompt_ids, MAX_TRANSLITERATE_TOKENS):
                            log.warning(
                                f"Skipping overlong mini-retry for '{display}' tags={list(tags)} "
                                f"({len(prompt_ids)} tokens)"
                            )
                            continue
                        mini_batch.append((display, tags, prompt_ids))
                    if mini_batch:
                        mini_outputs = llm.generate(
                            [_tokens_prompt(ids) for _d, _t, ids in mini_batch],
                            mini_params,
                            use_tqdm=False,
                        )
                        for (display, tags, _ids), output in zip(mini_batch, mini_outputs):
                            parsed = _parse_emit_scripts_output(output.outputs[0].text)
                            if parsed is None:
                                log.warning(f"No emit_scripts call on mini-retry for '{display}' (tags={list(tags)})")
                                continue
                            parsed = {t: v for t, v in parsed.items() if t in tags}
                            extra, _extra_rej = names_p1._filter_script_vars(
                                display, _expand_company_script_variants(parsed)
                            )
                            merged = script_vars_by_display.get(display, {})
                            for tag in tags:
                                if tag not in extra:
                                    continue
                                bucket = merged.setdefault(tag, [])
                                for val in extra[tag]:
                                    if val not in bucket:
                                        bucket.append(val)
                            script_vars_by_display[display] = merged

            written = 0
            for display, lat_vars in ready:
                script_vars = script_vars_by_display.get(display, {}) if not args.no_llm else {}
                print(_format_cluster(lat_vars, script_vars), file=out_fh)
                written += 1
            for line in passthrough:
                print(line, file=out_fh)
                written += 1

            log.info(f"Processed {min(batch_start + args.batch_size, len(lines))}/{len(lines)} ({written} written)")
    finally:
        if out_fh is not sys.stdout:
            out_fh.close()


if __name__ == "__main__":
    main()
