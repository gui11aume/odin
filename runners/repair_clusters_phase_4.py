"""Phase 4: repair malformed person-name clusters (flag -> LLM repair -> recompute).

Reads a phase-3-cleaned inventor cluster corpus (tab-separated ``tag{value}``
cells, one cluster per line) and writes back the full corpus in the exact same
format: lines that need no repair byte-identical, repaired lines fully
recomputed, unfixable lines dropped. Company corpora are out of scope.

Pipeline:

1. **Flag** (deterministic, CPU, whole corpus): parse each line, recover
   ``(given, family)`` from the first comma-form ``la{}`` cell (phase-2
   convention), and run a registry of detectors:

   * D1_title    — title/honorific tokens in any language (Dr, Prof, Ing,
     Dott, Mme, ...) in any position; trailing generational tokens excluded
   * D2_gen      — generational tokens (Jr/Sr/II/...) in non-trailing position
   * D3_particle — nobiliary particle on the given side of a comma form, or a
     lone-particle family side (short CJK-syllable particles suppressed when
     the cluster is CJK, so "Zu, Qun" is not mangled)
   * D4_noise    — digits, org/address/country lexicon, parens, "&"
   * D5_struct   — edge/multi commas, duplicated tokens, punctuation tokens
   * D6_taint    — Latin leakage into non-Latin cells; per-script title
     markers (Др/Д-Р, Δρ, ד׳ר, डॉ, ...)

   Ambiguous tokens (MS, Hon, ...) are flagged, never auto-classified: the LLM
   adjudicates. ``--no-llm`` stops after this stage (dry run: report + stats).

2. **Repair** (LLM, flagged clusters only, batched, one pointed retry):
   an ``audit_name`` tool call decides ``fix_name(given, family)`` /
   ``name_is_clean()`` / ``unrecoverable()`` — removal and edits are allowed,
   only tokens present in the cluster may survive; then phase 2's
   ``emit_alt_latin`` call re-runs on the corrected name so alt-Latin stays
   consistent.

3. **Recompute** (deterministic + phase-1 transliteration): the ``la`` side is
    re-expanded with ``latin_variants_with_alts``; the eleven non-Latin cells
    are re-transliterated with phase 1's stage-3/4 machinery (one
    all-tags ``emit_scripts`` call, per-missing-tag mini-retry, one full
    re-attempt) on the corrected seed; the line is reassembled with phase 1's
    assembler.

Drop criteria: no valid audit tool call after retry; repaired (given, family)
fails the full given+family gate; re-transliteration still incomplete after
mini-retry + one full re-attempt. Structurally unparseable lines pass through
unmodified.

Each LLM wave appends its records to a checkpoint file (``<output>.ckpt.jsonl``)
so an interrupted run can be continued with ``--resume``; the checkpoint is
removed on clean completion.

The model is served with ``vllm serve`` (tool calling enabled) and called over
the OpenAI-compatible API; every request carries ``--seed`` so runs are
reproducible.

Usage:
    python runners/repair_clusters_phase_4.py /mnt/nvme1/odin_train_set.clean.txt.gz \
        [-o /mnt/nvme1/odin_train_set.repaired.txt.gz] \
        [--base-url http://localhost:8010/v1] [--model Qwen/Qwen3.8-27B-FP8] \
        [--concurrency 16] [--wave-size 256] [--seed 42] [--no-llm] [--resume]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import io
import json
import logging
import re
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

# Allow `python runners/repair_clusters_phase_4.py` to import the shared core
# and the phase 1/2 machinery (reused, not reimplemented).
_RUNNERS = Path(__file__).resolve().parent
if str(_RUNNERS) not in sys.path:
    sys.path.insert(0, str(_RUNNERS))

import generate_names_core as core  # noqa: E402
import generate_names_phase_1 as p1  # noqa: E402
import generate_names_phase_2 as p2  # noqa: E402

log = logging.getLogger(__name__)

PROMPT_VERSION = "phase4-audit/1"

DEFAULT_MODEL = "Qwen/Qwen3.8-27B-FP8"
DEFAULT_BASE_URL = "http://localhost:8010/v1"

MAX_LA_IN_PROMPT = 24
MAX_FLAGS_IN_PROMPT = 12

MAX_AUDIT_TOKENS = 192
MAX_ALT_TOKENS = 192


# --------------------------------------------------------------------------- #
# Flags
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Flag:
    detector: str
    evidence: str


@dataclass
class ClusterView:
    """One parsed cluster line plus its stage-A findings."""

    line_no: int
    line: str
    cells: list[tuple[str, str]]
    la_forms: list[str]
    given: str
    family: str
    flags: list[Flag] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Stage A — detectors (pure functions; each returns a list of Flags)
# --------------------------------------------------------------------------- #
# Title/honorific surface forms, lowercased with a trailing period stripped.
# Single-letter dotted tokens ("M.") are initials, not titles, and are excluded
# by the initialism-shape check in detect_titles.
_TITLE_TOKENS = frozenset(
    {
        # English
        "dr",
        "prof",
        "mr",
        "mrs",
        "ms",
        "rev",
        "fr",
        "sir",
        "dame",
        "col",
        "capt",
        "cmdr",
        "hon",
        "phd",
        "ph.d",
        "md",
        "m.d",
        "esq",
        "st",
        # Spanish / Latin American
        "ing",
        "arq",
        "lic",
        "abg",
        "sr",
        "sra",
        # German
        "arch",
        "herr",
        "frau",
        "dipl",
        # French
        "mme",
        "mlle",
        "me",
        # Italian
        "dott",
        "dottore",
        "avv",
        # Portuguese / general
        "eng",
    }
)

_GENERATIONAL_KEYS = frozenset({"jr", "sr", "junior", "senior", "ii", "iii", "iv"})

# Particle tokens that double as CJK name syllables (Zu, Du, ...) — the CJK
# guard suppresses these only in the unambiguous single-token shapes.
_SHORT_CJK_PARTICLES = frozenset({"zu", "du", "di", "da", "do", "de", "te", "la", "le", "y", "del"})

_NOISE_TOKENS = frozenset(
    {
        "inc",
        "corp",
        "corporation",
        "ltd",
        "llc",
        "plc",
        "gmbh",
        "sarl",
        "ag",
        "bv",
        "nv",
        "sl",
        "spa",
        "sml",
        "university",
        "college",
        "institute",
        "laboratory",
        "lab",
        "bldg",
        "building",
        "avenue",
        "blvd",
        "street",
        "road",
        "lane",
        "suite",
        "apt",
        "campus",
        "plaza",
        # NB: "park" is not a noise word — it is a common Korean surname.
        "usa",
        "us",
        "u.s.a",
        "uk",
        "canada",
        "france",
        "germany",
        "india",
        "china",
        "japan",
        "korea",
        "mexico",
        "brazil",
        "sweden",
        "italy",
        "spain",
        "russia",
        "poland",
        "denmark",
        "norway",
        "finland",
        "austria",
        "switzerland",
        "netherlands",
        "belgium",
        "portugal",
        "greece",
        "turkey",
        "israel",
        "singapore",
        "australia",
        "new",
        "zealand",
    }
)

# Per-script title markers (token-level match; substring matching would hit
# legitimate Cyrillic/Greek syllables like Драгомир).
_TITLE_MARKERS: dict[str, frozenset[str]] = {
    "cy": frozenset({"др", "д-р", "д р", "доктор"}),
    "gk": frozenset({"δρ"}),
    # NB: the single letter "د" is NOT a marker — it matches Arabic initials
    # ("د."); a "Dr." taint on the Latin side is already caught by D1, since
    # every script cell is generated from the same la seed.
    "ab": frozenset({"دكتور"}),
    "dv": frozenset({"डॉ", "डॉक्टर"}),
    "hb": frozenset({"ד׳ר", "ד״ר"}),
    "th": frozenset({"ดร"}),
    "gg": frozenset({"დრ", "დ-რ"}),
    "am": frozenset({"դր", "դ-ր"}),
}

_LATIN_LEAK = re.compile(r"[A-Za-z]")
_INITIALISM = re.compile(r"(?:[A-Za-z]\.)+")
_PUNCT_ONLY = re.compile(r"[.,'\-]+")
_HAN = re.compile(r"[\u4E00-\u9FFF]")


def _token_key(tok: str) -> str:
    return tok.lower().rstrip(".")


def _position(i: int, n: int) -> str:
    if i == 0:
        return "lead"
    if i == n - 1:
        return "trail"
    return "middle"


def detect_titles(la_forms: list[str]) -> list[Flag]:
    flags: list[Flag] = []
    seen: set[tuple[str, str]] = set()
    for form in la_forms:
        toks = form.split()
        for i, tok in enumerate(toks):
            if _INITIALISM.fullmatch(tok):
                continue
            k = _token_key(tok)
            if len(k) < 2 or k not in _TITLE_TOKENS:
                continue
            if i == len(toks) - 1 and k in _GENERATIONAL_KEYS:
                continue  # trailing Jr/Sr/II/... is the generational design
            pos = _position(i, len(toks))
            key = (k, pos)
            if key in seen:
                continue
            seen.add(key)
            flags.append(Flag("D1_title", f"'{tok}' ({pos}) in '{form}'"))
    return flags


def detect_generational_misplaced(la_forms: list[str]) -> list[Flag]:
    flags: list[Flag] = []
    seen: set[tuple[str, str]] = set()
    for form in la_forms:
        toks = form.split()
        for i, tok in enumerate(toks):
            k = _token_key(tok)
            if k not in _GENERATIONAL_KEYS or i == len(toks) - 1:
                continue
            pos = _position(i, len(toks))
            key = (k, pos)
            if key in seen:
                continue
            seen.add(key)
            flags.append(Flag("D2_generational", f"'{tok}' ({pos}) in '{form}'"))
    return flags


def _has_cjk(cells: list[tuple[str, str]]) -> bool:
    return any(tag != "la" and _HAN.search(val) for tag, val in cells)


def detect_particles(cells: list[tuple[str, str]], la_forms: list[str]) -> list[Flag]:
    cjk = _has_cjk(cells)
    flags: list[Flag] = []
    seen: set[str] = set()
    for form in la_forms:
        if "," not in form:
            continue
        fam, _, given = form.partition(",")
        fam_toks = fam.split()
        given_toks = given.split()
        if len(fam_toks) == 1 and not core._is_initial_only_token(fam_toks[0]):
            k = _token_key(fam_toks[0])
            if k in core._PARTICLES and not (cjk and k in _SHORT_CJK_PARTICLES):
                if k not in seen:
                    seen.add(k)
                    flags.append(Flag("D3_particle_family", f"'{fam_toks[0]}' lone-particle family side in '{form}'"))
        for t in given_toks:
            if core._is_initial_only_token(t):
                continue  # "Y." is an initial, not the particle "y"
            k = _token_key(t)
            if k not in core._PARTICLES:
                continue
            if cjk and k in _SHORT_CJK_PARTICLES and len(given_toks) == 1:
                continue
            if k not in seen:
                seen.add(k)
                flags.append(Flag("D3_particle_given", f"'{t}' on given side in '{form}'"))
    return flags


def detect_noise(la_forms: list[str]) -> list[Flag]:
    flags: list[Flag] = []
    seen: set[str] = set()
    for form in la_forms:
        if any(c.isdigit() for c in form):
            if "D4_digit" not in seen:
                seen.add("D4_digit")
                flags.append(Flag("D4_digit", f"digit in '{form}'"))
        if "(" in form or ")" in form:
            if "D4_paren" not in seen:
                seen.add("D4_paren")
                flags.append(Flag("D4_paren", f"parentheses in '{form}'"))
        if "&" in form:
            if "D4_amp" not in seen:
                seen.add("D4_amp")
                flags.append(Flag("D4_amp", f"'&' in '{form}'"))
        for tok in form.split():
            k = _token_key(tok)
            if k in _NOISE_TOKENS and k not in seen:
                seen.add(k)
                flags.append(Flag("D4_noise", f"'{tok}' in '{form}'"))
    return flags


def detect_structural(la_forms: list[str]) -> list[Flag]:
    flags: list[Flag] = []
    seen: set[str] = set()
    for form in la_forms:
        v = form.strip()
        if (v.startswith(",") or v.endswith(",")) and "D5_edge_comma" not in seen:
            seen.add("D5_edge_comma")
            flags.append(Flag("D5_edge_comma", f"edge comma in '{form}'"))
        if v.count(",") > 1 and "D5_multi_comma" not in seen:
            seen.add("D5_multi_comma")
            flags.append(Flag("D5_multi_comma", f"multiple commas in '{form}'"))
        toks = v.split()
        for a, b in zip(toks, toks[1:]):
            # Repeated initials (two middle initials: "A. A.") are legitimate.
            if (
                a.lower() == b.lower()
                and not (core._is_initial_only_token(a) and core._is_initial_only_token(b))
                and a.lower() not in seen
            ):
                seen.add(a.lower())
                flags.append(Flag("D5_duplicate", f"'{a} {b}' in '{form}'"))
        for tok in toks:
            if _PUNCT_ONLY.fullmatch(tok) and "D5_punct" not in seen:
                seen.add("D5_punct")
                flags.append(Flag("D5_punct", f"punctuation-only token '{tok}' in '{form}'"))
    return flags


def detect_nonlatin_taint(cells: list[tuple[str, str]]) -> list[Flag]:
    flags: list[Flag] = []
    for tag, val in cells:
        if tag == "la":
            continue
        if _LATIN_LEAK.search(val):
            flags.append(Flag("D6_latin_leak", f"latin chars in {tag}{{{val[:40]}}}"))
            continue
        markers = _TITLE_MARKERS.get(tag)
        if not markers:
            continue
        toks = {t.lower().rstrip(".") for t in val.split()}
        for m in sorted(markers):
            if m.rstrip(".") in toks:
                flags.append(Flag("D6_title_taint", f"'{m}' in {tag} cell"))
    return flags


DETECTORS: list[tuple[str, Any]] = [
    ("D1_title", detect_titles),
    ("D2_generational", detect_generational_misplaced),
    ("D3_particle", detect_particles),
    ("D4_noise", detect_noise),
    ("D5_structural", detect_structural),
    ("D6_taint", detect_nonlatin_taint),
]


def flag_cluster(line_no: int, line: str, cells: list[tuple[str, str]], la_forms: list[str]) -> ClusterView:
    given, family = core.given_family_from_la_forms(la_forms)
    view = ClusterView(line_no=line_no, line=line, cells=cells, la_forms=la_forms, given=given, family=family)
    for name, fn in DETECTORS:
        if name == "D3_particle":
            view.flags.extend(fn(cells, la_forms))
        else:
            view.flags.extend(fn(la_forms) if name != "D6_taint" else fn(cells))
    return view


# --------------------------------------------------------------------------- #
# Stage B — LLM client + audit prompt
# --------------------------------------------------------------------------- #
AUDIT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "fix_name",
            "description": "The parsed name has a real defect. Emit the corrected given/family split.",
            "parameters": {
                "type": "object",
                "properties": {
                    "given": {
                        "type": "string",
                        "description": (
                            "Corrected given name(s): title-case full words; initials as a single "
                            "uppercase letter + period. Empty string only when there is no given name."
                        ),
                    },
                    "family": {
                        "type": "string",
                        "description": "Corrected family name(s): title-case; preserve hyphens and spaces.",
                    },
                },
                "required": ["given", "family"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "name_is_clean",
            "description": "The name is well-formed; the flag is a false positive. No change.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "unrecoverable",
            "description": "No valid given+family pair can be recovered from the evidence.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

AUDIT_SYSTEM_PROMPT = (
    "You are a name-repair auditor for a cross-script person-name corpus. "
    "Call exactly one tool: fix_name, name_is_clean, or unrecoverable. No free text."
)

_AUDIT_CONTRACT = """\
Audit one person-name cluster from a cross-script name-variant corpus. The \
cluster holds surface forms of ONE person in twelve scripts. A parser recovered \
(given, family) from the Latin forms; a deterministic pre-filter flagged possible \
defects. Decide exactly one tool call.

### Defects and their fixes
1. Titles / honorifics in ANY language are not name parts — remove them, and \
never treat one as an initial: Dr/Prof/Mr/Mrs/Ms/Rev/Fr/Sir/Dame/Col/Capt/Hon/PhD/\
MD/Esq/St (English); Ing/Arq/Lic/Abg/Sr/Sra (Spanish/Latin American); Dr/Prof/Ing/\
Arch/Dipl (German); M/Mme/Mlle/Me (French); Dott/Dottore/Avv/Ing (Italian); Dr/Ir \
(Dutch); Dr/Eng (Portuguese); and equivalents from other languages. \
"Peter Dr. Flury" -> given="Peter", family="Flury". "Flury Dr." -> family="Flury".
2. Generational markers (Jr/Junior/Sr/Senior/II/III/IV) belong ONLY as a trailing \
family suffix. If one appears elsewhere, move it to the family tail \
("II John Smith" -> given="John", family="Smith II") or remove it if it is not \
generational.
3. Nobiliary particles (de/di/da/van/von/der/den/des/te/ten/zu/del/della/degli/...) \
belong to the FAMILY name. Re-attach when split off: "Dyke, Kelly Van" -> \
given="Kelly", family="Van Dyke". EXCEPTION: in CJK-romanized names such tokens \
are usually genuine name syllables (Zu, Du, Di, La, Le, Da, Do, Te, De, Y) — the \
non-Latin cells tell you the script; never break a CJK romanization ("Zu, Qun" is \
a Chinese name; leave it).
4. Non-name noise — remove: addresses/streets, organizations (Inc/Corp/Ltd/...), \
digits, country names, parentheticals, "et al".
5. Obvious OCR residue in a name token (5->S, 0->O, 1->l/I) — fix when unambiguous.
6. Wrong given/family boundary or swap — correct it using the surface forms as \
evidence (multi-token families stay whole: "Lopez Fernandez, Ivan" keeps \
family="Lopez Fernandez").
7. Duplicated tokens ("Smith Smith") — deduplicate.

### Hard constraints
- Use ONLY words/letters already present in the Latin surface forms. Never invent, \
guess, or expand anything (never expand an initial into a full name).
- Keep middle initials. Keep diacritics exactly as written. Output Title Case.
- When a flagged token is ambiguous (MS/Hon/St that could be a real name or \
initials), decide from the whole cluster; when it reads as a name, call \
name_is_clean().
- When in doubt between fix and clean, prefer name_is_clean() — do not "improve" \
a plausible name.

### Input
Parsed: given="{given}", family="{family}"
Latin surface forms:
{la_forms}
Non-Latin samples (first form per script):
{nonlatin}
Flagged issues:
{flags}
"""


def make_audit_user_message(view: ClusterView) -> str:
    la_lines = "\n".join(f"  {f}" for f in view.la_forms[:MAX_LA_IN_PROMPT])
    if len(view.la_forms) > MAX_LA_IN_PROMPT:
        la_lines += f"\n  ... ({len(view.la_forms) - MAX_LA_IN_PROMPT} more)"
    samples = []
    for tag, _ in view.cells:
        if tag != "la":
            val = next((v for t, v in view.cells if t == tag), "")
            samples.append(f"  {tag}: {val}")
    flag_lines = "\n".join(f"  {f.detector}: {f.evidence}" for f in view.flags[:MAX_FLAGS_IN_PROMPT])
    if len(view.flags) > MAX_FLAGS_IN_PROMPT:
        flag_lines += f"\n  ... ({len(view.flags) - MAX_FLAGS_IN_PROMPT} more)"
    return _AUDIT_CONTRACT.format(
        given=view.given,
        family=view.family,
        la_forms=la_lines,
        nonlatin="\n".join(samples) if samples else "  (none)",
        flags=flag_lines,
    )


def make_audit_retry_user_message(view: ClusterView) -> str:
    return make_audit_user_message(view) + "\n\nYou MUST respond with exactly one tool call. No free text."


class RepairLLM:
    """Thin wrapper over the OpenAI-compatible vLLM server (tool calls)."""

    def __init__(self, base_url: str, model: str, seed: int):
        self.client = OpenAI(base_url=base_url, api_key="unused")  # nosec B104  # local server, no auth
        self.model = model
        self.seed = seed
        self.n_calls = 0

    def call(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        tool_choice: str = "auto",
    ) -> tuple[str, dict[str, Any]] | None:
        """One chat completion expecting exactly one tool call.

        Returns ``(tool_name, args)`` or None when no tool call came back.
        Retries transient API errors; raises on persistent failure.
        """
        last: Exception | None = None
        for attempt in range(5):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    tool_choice=tool_choice,
                    seed=self.seed,
                    # Phase 1 ran with thinking off; with the server's
                    # reasoning parser on, thinking otherwise eats the token
                    # budget and no tool call comes back.
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                self.n_calls += 1
                msg = resp.choices[0].message
                if not msg.tool_calls:
                    return None
                tc = msg.tool_calls[0]
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    return None
                if not isinstance(args, dict):
                    return None
                return tc.function.name, args
            except (APIConnectionError, APITimeoutError, APIError) as exc:  # retry transient errors
                last = exc
                time.sleep(2.0 * (2**attempt))
        raise RuntimeError(f"LLM call failed after retries: {last}") from last


# --------------------------------------------------------------------------- #
# Stage B — per-cluster repair (audit + alt-Latin)
# --------------------------------------------------------------------------- #
def audit_cluster(llm: RepairLLM, view: ClusterView) -> tuple[str, str | None, str | None]:
    """Run the audit agent (with one pointed retry).

    Returns ``(outcome, given, family)`` where outcome is "fixed", "clean", or
    "dropped"; given/family are set only for "fixed".
    """
    for user_msg, choice in (
        (make_audit_user_message(view), "auto"),
        (make_audit_retry_user_message(view), "required"),
    ):
        res = llm.call(
            [{"role": "system", "content": AUDIT_SYSTEM_PROMPT}, {"role": "user", "content": user_msg}],
            AUDIT_TOOLS,
            temperature=0.0,
            top_p=1.0,
            max_tokens=MAX_AUDIT_TOKENS,
            tool_choice=choice,
        )
        name, args = res if res is not None else (None, None)
        if name == "name_is_clean":
            return "clean", None, None
        if name == "unrecoverable":
            return "dropped", None, None
        if name == "fix_name":
            given = str((args or {}).get("given", "")).strip()
            family = str((args or {}).get("family", "")).strip()
            if core.is_usable_person_name(given, family):
                return "fixed", given, family
            return "dropped", None, None
    return "dropped", None, None


def alt_latin_cluster(llm: RepairLLM, label: str) -> list[str]:
    """Phase 2's emit_alt_latin call on the corrected display name."""
    base = [
        {"role": "system", "content": p2.ALT_LATIN_SYSTEM_PROMPT},
    ]
    res = llm.call(
        base + [{"role": "user", "content": p2._ALT_LATIN_USER_CONTRACT + label}],  # noqa: SLF001
        p2.ALT_LATIN_TOOLS,
        temperature=0.2,
        top_p=0.95,
        max_tokens=MAX_ALT_TOKENS,
        tool_choice="auto",
    )
    name, args = res if res is not None else (None, None)
    if name != "emit_alt_latin":
        res = llm.call(
            base + [{"role": "user", "content": p2._make_alt_latin_retry_user_message(label)}],  # noqa: SLF001
            p2.ALT_LATIN_TOOLS,
            temperature=0.5,
            top_p=0.95,
            max_tokens=MAX_ALT_TOKENS,
            tool_choice="required",
        )
        name, args = res if res is not None else (None, None)
    if name != "emit_alt_latin":
        return []
    return core.normalize_string_list((args or {}).get("alts", []), core.MAX_ALT_LATIN)


# --------------------------------------------------------------------------- #
# Stage C — re-transliteration (phase 1 machinery), wave-level
# --------------------------------------------------------------------------- #
def _emit_batch(
    llm: RepairLLM,
    items: list[tuple[int, str, tuple[str, ...]]],
    *,
    temperature: float,
    concurrency: int,
) -> list[tuple[int, dict[str, list[str]]]]:
    """Batched emit_scripts calls. ``items`` are ``(idx, label, tags)`` with
    per-item tool sets; order is preserved. Returns one ``(idx, {tag:
    [forms]})`` per item (empty dict on no/foreign tool call)."""

    def one(item: tuple[int, str, tuple[str, ...]]) -> tuple[int, dict[str, list[str]]]:
        idx, label, tags = item
        res = llm.call(
            [
                {"role": "system", "content": p1._xlit_system_prompt(tags)},  # noqa: SLF001
                {"role": "user", "content": p1._make_xlit_user_message(label, tags)},  # noqa: SLF001
            ],
            p1._emit_scripts_tools(tags),  # noqa: SLF001
            temperature=temperature,
            top_p=0.95,
            max_tokens=p1.MAX_TRANSLITERATE_TOKENS,
            tool_choice="auto",
        )
        name, args = res if res is not None else (None, None)
        if name != "emit_scripts":
            return idx, {}
        parsed = p1._dispatch_emit_scripts(args)  # noqa: SLF001
        return idx, {t: v for t, v in parsed.items() if t in tags}

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(one, items))


def transcribe_wave(
    llm: RepairLLM,
    todo: dict[int, str],
    concurrency: int,
) -> dict[int, dict[str, list[str]]]:
    """Re-transliterate ``{idx: corrected seed label}`` into all eleven scripts.

    One emit_scripts call per cluster covering all eleven tags (phase 1 split
    the work over three passes for its offline batch API; over the serving API
    a single call is strictly cheaper), then per-missing-tag mini-retry, then
    one full re-attempt (a single call per cluster covering only its
    still-missing tags). LLM seeds accumulate in ``raw``; phase 1's expand+
    filter runs over the raw seeds only (as in phase 1, expansion is never
    applied to already-expanded output). Returns idx -> filtered script_vars;
    callers check ``p1._missing_scripts`` for drops.
    """
    raw: dict[int, dict[str, list[str]]] = {i: {} for i in todo}
    vars_: dict[int, dict[str, list[str]]] = {i: {} for i in todo}
    missing: dict[int, list[str]] = {i: list(p1.SCRIPTS) for i in todo}  # noqa: SLF001

    def refilter(i: int) -> None:
        vars_[i], _rejected = p1._filter_script_vars(todo[i], p1._expand_script_variants(raw[i]))  # noqa: SLF001
        missing[i] = p1._missing_scripts(vars_[i])  # noqa: SLF001

    # --- one full pass: all eleven tags in a single call per cluster ---
    all_tags = tuple(p1.SCRIPTS)  # noqa: SLF001
    for idx, parsed in _emit_batch(
        llm, [(i, todo[i], all_tags) for i in todo], temperature=0.2, concurrency=concurrency
    ):
        raw[idx] = p1._merge_script_vars(raw[idx], parsed)  # noqa: SLF001
    for i in todo:
        refilter(i)

    # --- mini-retry: one narrow call per missing tag ---
    for idx, parsed in _emit_batch(
        llm,
        [(i, todo[i], (tag,)) for i, tags in missing.items() for tag in tags],
        temperature=0.7,
        concurrency=concurrency,
    ):
        if parsed:
            raw[idx] = p1._merge_script_vars(raw[idx], parsed)  # noqa: SLF001
    for i in todo:
        if missing[i]:
            refilter(i)

    # --- one full re-attempt: single call per cluster for its missing tags ---
    for idx, parsed in _emit_batch(
        llm,
        [(i, todo[i], tuple(missing[i])) for i, tags in missing.items() if tags],
        temperature=0.2,
        concurrency=concurrency,
    ):
        if parsed:
            raw[idx] = p1._merge_script_vars(raw[idx], parsed)  # noqa: SLF001
    for i in todo:
        if missing[i]:
            refilter(i)

    return vars_


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def iter_lines(path: Path) -> Iterator[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            yield line


def open_out(path: Path):
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.GzipFile(path, mode="wb", mtime=0), encoding="utf-8")
    return open(path, "wt", encoding="utf-8")


def default_paths(inp: Path) -> tuple[Path, Path, Path]:
    if inp.name.endswith(".txt.gz"):
        stem, suf = inp.name[: -len(".txt.gz")], ".txt.gz"
    elif inp.suffix:
        stem, suf = inp.name[: -len(inp.suffix)], inp.suffix
    else:
        stem, suf = inp.name, ""
    out = inp.with_name(stem + ".repaired" + suf)
    return out, out.with_name(stem + ".repaired.repairs.jsonl"), out.with_name(stem + ".repaired.meta.json")


def ckpt_path_for(out: Path) -> Path:
    return out.with_name(out.name + ".ckpt.jsonl")


def load_checkpoint(path: Path) -> dict[int, tuple[str, str | None, dict[str, Any]]]:
    """Read a wave checkpoint into {line_no: (outcome, new_line, record)}.

    A torn final line (crash mid-append) is skipped; its cluster is simply
    re-processed on resume.
    """
    results: dict[int, tuple[str, str | None, dict[str, Any]]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                log.warning("Skipping torn checkpoint line in %s", path)
                continue
            results[rec["line_no"]] = (rec["outcome"], rec.get("new_line"), rec)
    return results


def append_checkpoint(path: Path, records: list[dict[str, Any]]) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        for rec in records:
            json.dump(rec, fh, ensure_ascii=False)
            fh.write("\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4: repair malformed person-name clusters.")
    parser.add_argument("input", help="Phase-3-cleaned cluster corpus (.txt or .txt.gz), one cluster per line.")
    parser.add_argument("-o", "--output", default=None, help="Output corpus path (default: <input>.repaired[.ext]).")
    parser.add_argument("--report", default=None, help="Repair report JSONL path.")
    parser.add_argument("--meta", default=None, help="Stats meta JSON path.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="vLLM OpenAI-compatible base URL.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Served model name.")
    parser.add_argument("--concurrency", type=int, default=16, help="Concurrent LLM requests.")
    parser.add_argument("--wave-size", type=int, default=256, help="Flagged clusters processed per LLM wave.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed for reproducible runs.")
    parser.add_argument(
        "--no-llm", action="store_true", help="Dry run: flag + report + stats, no LLM, no output corpus."
    )
    parser.add_argument("--max-flagged", type=int, default=0, help="Debug: cap on LLM-repaired clusters (0 = all).")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from an existing wave checkpoint (one exists at <output>.ckpt.jsonl).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)

    inp = Path(args.input)
    if not inp.is_file():
        raise SystemExit(f"Input not found: {inp}")
    def_out, def_report, def_meta = default_paths(inp)
    out = Path(args.output) if args.output else def_out
    report_path = Path(args.report) if args.report else def_report
    meta_path = Path(args.meta) if args.meta else def_meta

    # ---------------- Stage A: flag ---------------- #
    t0 = time.time()
    flagged: list[ClusterView] = []
    n_lines = 0
    n_unparseable = 0
    n_no_la = 0
    flags_by_detector: dict[str, int] = {}
    for line_no, line in enumerate(iter_lines(inp), start=1):
        n_lines += 1
        cells = core.parse_cluster_line(line.rstrip("\r\n"))
        if cells is None:
            n_unparseable += 1
            continue
        la_forms = [v for t, v in cells if t == "la"]
        if not la_forms:
            n_no_la += 1
            continue
        view = flag_cluster(line_no, line, cells, la_forms)
        if view.flags:
            for f in view.flags:
                top = f.detector.split("_")[0]
                flags_by_detector[top] = flags_by_detector.get(top, 0) + 1
            flagged.append(view)
        if n_lines % 500_000 == 0:
            log.info("scanned %d lines (%d flagged so far)", n_lines, len(flagged))
    log.info(
        "Stage A: %d lines, %d flagged, %d unparseable, %d no-la; flags by detector: %s (%.1fs)",
        n_lines,
        len(flagged),
        n_unparseable,
        n_no_la,
        flags_by_detector,
        time.time() - t0,
    )

    if args.no_llm:
        _write_dry_report(report_path, flagged)
        _write_meta(
            meta_path,
            {
                "input": str(inp),
                "output": None,
                "model": None,
                "seed": None,
                "mode": "dry-run",
                "n_lines": n_lines,
                "n_flagged": len(flagged),
                "n_unparseable": n_unparseable,
                "n_no_la": n_no_la,
                "flags_by_detector": flags_by_detector,
                "seconds": round(time.time() - t0, 2),
            },
        )
        return

    # ---------------- Stage B+C: repair + recompute (waves) ---------------- #
    ckpt_path = ckpt_path_for(out)
    if ckpt_path.exists() and not args.resume:
        raise SystemExit(
            f"Wave checkpoint exists: {ckpt_path}. Pass --resume to continue, or delete it to start fresh."
        )

    # line_no -> (outcome, new_line|None, record)
    results: dict[int, tuple[str, str | None, dict[str, Any]]] = {}
    n_repaired = n_clean = n_dropped = 0
    dropped_by_reason: dict[str, int] = {}
    if args.resume and ckpt_path.exists():
        results = load_checkpoint(ckpt_path)
        for outcome, _new_line, rec in results.values():
            if outcome == "fixed":
                n_repaired += 1
            elif outcome == "clean":
                n_clean += 1
            else:
                n_dropped += 1
                reason = rec.get("drop_reason") or "unspecified"
                dropped_by_reason[reason] = dropped_by_reason.get(reason, 0) + 1
        log.info("Resuming: %d clusters already resolved from %s", len(results), ckpt_path)
    elif args.resume:
        log.warning("--resume given but no checkpoint at %s; starting fresh", ckpt_path)

    capped = len(flagged) if args.max_flagged <= 0 else min(len(flagged), args.max_flagged)
    to_repair = [v for v in flagged[:capped] if v.line_no not in results]
    if args.max_flagged > 0 and len(flagged) > capped:
        log.warning(
            "--max-flagged active: only %d of %d flagged clusters will be LLM-repaired; the rest pass through unmodified",
            capped,
            len(flagged),
        )

    llm = RepairLLM(args.base_url, args.model, args.seed)

    for wave_start in range(0, len(to_repair), args.wave_size):
        wave = to_repair[wave_start : wave_start + args.wave_size]
        t_wave = time.time()
        log.info("wave %d: %d clusters (audit)", wave_start // args.wave_size + 1, len(wave))

        # --- audit ---
        def one_audit(view: ClusterView) -> tuple[int, tuple[str, str | None, str | None]]:
            return view.line_no, audit_cluster(llm, view)

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            audits = dict(pool.map(one_audit, wave))
        # --- alt-Latin for fixed clusters ---
        fixed = [(v, audits[v.line_no]) for v in wave if audits[v.line_no][0] == "fixed"]
        alts: dict[int, list[str]] = {}
        if fixed:

            def one_alt(v_label: tuple[ClusterView, tuple[str, str | None, str | None]]) -> tuple[int, list[str]]:
                v, (_o, g2, f2) = v_label
                return v.line_no, alt_latin_cluster(llm, core.display_name(g2 or "", f2 or ""))

            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                alts = dict(pool.map(one_alt, fixed))

        # --- transcribe fixed clusters (wave-level, parallel) ---
        script_vars_all: dict[int, dict[str, list[str]]] = {}
        if fixed:
            todo = {v.line_no: core.display_name(g2 or "", f2 or "") for v, (_o, g2, f2) in fixed}
            script_vars_all = transcribe_wave(llm, todo, args.concurrency)

        # --- assemble ---
        for v in wave:
            outcome, g2, f2 = audits[v.line_no]
            warnings: list[str] = []
            new_line: str | None = None
            drop_reason = "unspecified"
            if outcome == "fixed":
                g2, f2 = g2 or "", f2 or ""
                if (core.title_name(g2), core.title_name(f2)) == (core.title_name(v.given), core.title_name(v.family)):
                    outcome = "clean"  # no-op fix
                    g2 = f2 = None
                else:
                    still = p1._missing_scripts(script_vars_all.get(v.line_no, {}))  # noqa: SLF001
                    if still:
                        outcome = "dropped"
                        drop_reason = "transliteration_incomplete"
                        warnings.append(f"transliteration incomplete: {still}")
                        g2 = f2 = None
                    else:
                        la = core.latin_variants_with_alts(g2, f2, alts.get(v.line_no, []))
                        new_line = p1._format_cluster(la, script_vars_all[v.line_no])  # noqa: SLF001
            elif outcome == "dropped":
                drop_reason = "audit_unrecoverable_or_invalid"
                warnings.append("audit: unrecoverable or no valid fix")
            record = {
                "line_no": v.line_no,
                "original": v.line.rstrip("\r\n"),
                "flags": [{"d": f.detector, "e": f.evidence} for f in v.flags],
                "given": v.given,
                "family": v.family,
                "fixed_given": g2,
                "fixed_family": f2,
                "alts": alts.get(v.line_no) if outcome == "fixed" else None,
                "new_line": new_line,
                "outcome": outcome,
                "drop_reason": drop_reason if outcome == "dropped" else None,
                "warnings": warnings,
            }
            results[v.line_no] = (outcome, new_line, record)
            if outcome == "fixed":
                n_repaired += 1
            elif outcome == "clean":
                n_clean += 1
            else:
                n_dropped += 1
                dropped_by_reason[drop_reason] = dropped_by_reason.get(drop_reason, 0) + 1
        append_checkpoint(ckpt_path, [results[v.line_no][2] for v in wave])
        log.info(
            "wave done in %.1fs (audit=%d alt=%d xlit>=%d)",
            time.time() - t_wave,
            len(wave),
            len(fixed),
            len(fixed),
        )

    # ---------------- write output (order-preserving) + report + meta ---------------- #
    report_fh = open(report_path, "w", encoding="utf-8")
    out_fh = open_out(out)
    written = 0
    try:
        for line_no, line in enumerate(iter_lines(inp), start=1):
            res = results.get(line_no)
            if res is None:
                out_fh.write(line if line.endswith("\n") else line + "\n")
                written += 1
                continue
            outcome, new_line, record = res
            json.dump(record, report_fh, ensure_ascii=False)
            report_fh.write("\n")
            if outcome == "dropped":
                continue
            out_fh.write((line if line.endswith("\n") else line + "\n") if outcome == "clean" else new_line + "\n")
            written += 1
    finally:
        report_fh.close()
        out_fh.close()

    meta = {
        "input": str(inp),
        "output": str(out),
        "model": args.model,
        "seed": args.seed,
        "prompt_version": PROMPT_VERSION,
        "n_lines": n_lines,
        "n_unparseable": n_unparseable,
        "n_no_la": n_no_la,
        "n_flagged": len(flagged),
        "n_repaired": n_repaired,
        "n_clean_adjudicated": n_clean,
        "n_dropped": n_dropped,
        "dropped_by_reason": dropped_by_reason,
        "n_output_lines": written,
        "flags_by_detector": flags_by_detector,
        "n_llm_calls": llm.n_calls,
        "seconds": round(time.time() - t0, 2),
    }
    _write_meta(meta_path, meta)
    ckpt_path.unlink(missing_ok=True)  # clean completion: checkpoint no longer needed
    log.info(
        "Phase 4 done: %d lines -> %d written (%d repaired, %d clean, %d dropped) in %.1fs",
        n_lines,
        written,
        n_repaired,
        n_clean,
        n_dropped,
        time.time() - t0,
    )


def _write_dry_report(path: Path, flagged: list[ClusterView]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for v in flagged:
            json.dump(
                {
                    "line_no": v.line_no,
                    "original": v.line.rstrip("\r\n"),
                    "given": v.given,
                    "family": v.family,
                    "flags": [{"d": f.detector, "e": f.evidence} for f in v.flags],
                },
                fh,
                ensure_ascii=False,
            )
            fh.write("\n")
    log.info("dry-run report: %d flagged clusters -> %s", len(flagged), path)


def _write_meta(path: Path, meta: dict[str, Any]) -> None:
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log.info("meta -> %s", path)


if __name__ == "__main__":
    main()
