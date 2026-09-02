"""Shared core for name-variant generation (phase 1 + phase 2).

Owns model/vLLM helpers, Latin variant expansion, name-extraction tools, and
tool-call parsing. Phase scripts import from here and keep their own
transliteration / alt-Latin pipelines.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
import unicodedata
from itertools import product
from typing import Any, Iterator

from vllm import LLM, SamplingParams, TokensPrompt

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3.6-35B-A3B-FP8"
# Extract and transliterate prompts (system+tools+user contract) fit within
# MAX_MODEL_LEN together with the completion and pointed retry messages.
MAX_MODEL_LEN = 4096
MAX_EXTRACT_TOKENS = 192
MAX_NUM_SEQS = 64
# Optional common-romanization alts (phase 2); cap enforced in Python.
MAX_ALT_LATIN = 3

SCRIPTS = ["cy", "gk", "ab", "cn", "jp", "kr", "dv", "hb", "th", "gg", "am"]

# ---------------------------------------------------------------------------
# Diacritic helpers
# ---------------------------------------------------------------------------

# German / Nordic multi-letter ASCII expansions (applied before strip).
_DIGRAPH = str.maketrans(
    {
        "ä": "ae",
        "Ä": "Ae",
        "ö": "oe",
        "Ö": "Oe",
        "ü": "ue",
        "Ü": "Ue",
        "ß": "ss",
        "ø": "oe",
        "Ø": "Oe",
    }
)

# Non-decomposing Latin letters that survive NFD mark-stripping.
_LATIN_ASCII = str.maketrans(
    {
        "ł": "l",
        "Ł": "L",
        "ø": "o",
        "Ø": "O",
        "æ": "ae",
        "Æ": "Ae",
        "ð": "d",
        "Ð": "D",
        "þ": "th",
        "Þ": "Th",
        "đ": "d",
        "Đ": "D",
        "ħ": "h",
        "Ħ": "H",
        "œ": "oe",
        "Œ": "Oe",
        "ı": "i",
        "ĳ": "ij",
        "Ĳ": "IJ",
    }
)


def strip_diacritics(text: str) -> str:
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return text.translate(_LATIN_ASCII)


def digraph(text: str) -> str:
    return strip_diacritics(text.translate(_DIGRAPH))


# ---------------------------------------------------------------------------
# Latin variant generator
# ---------------------------------------------------------------------------


def _is_initialism_token(tok: str) -> bool:
    """True for dotted/hyphenated initialisms: `E.`, `E.A.`, `J-M`, `J.-M.`."""
    return bool(
        re.fullmatch(r"(?:[A-Za-z]\.)+", tok)  # E. / E.A.
        or re.fullmatch(r"[A-Za-z](?:-[A-Za-z])+", tok)  # J-M
        or re.fullmatch(r"(?:[A-Za-z]\.-)+[A-Za-z]\.", tok)  # J.-M.
    )


def _is_initial_only_token(tok: str) -> bool:
    """True for a bare letter or initialism: `G`, `G.`, `E.A.`, `J-M`."""
    tok = tok.strip()
    if not tok:
        return False
    if len(tok) == 1 and tok.isalpha():
        return True
    return _is_initialism_token(tok)


def _side_has_full_name_token(text: str) -> bool:
    """True if at least one whitespace token is a real name word (not an initial)."""
    return any(not _is_initial_only_token(tok) for tok in text.split() if tok)


def is_usable_person_name(given: str, family: str) -> bool:
    """Require both a full given token and a full family token.

    Middle initials are fine (`Ernst A.` + `Mayr`). Mononyms, initial-only
    givens (`G` / `G.`), and initial-only families are not.
    """
    given, family = given.strip(), family.strip()
    if not given or not family:
        return False
    return _side_has_full_name_token(given) and _side_has_full_name_token(family)


def title_token(tok: str) -> str:
    """Title-case one whitespace-separated name token."""
    if not tok:
        return tok
    if _is_initialism_token(tok):
        return tok.upper()
    if "-" in tok:
        return "-".join(title_token(p) for p in tok.split("-"))
    # McMinn / McDonald: capitalize the letter after Mc (ALL-CAPS MCMINN → McMinn).
    lower = tok.lower()
    if lower.startswith("mc") and len(tok) > 2 and tok[2].isalpha():
        return "Mc" + tok[2].upper() + tok[3:].lower()
    return tok[0].upper() + tok[1:].lower()


def title_name(text: str) -> str:
    """Title-case a multi-token name fragment (given or family)."""
    return " ".join(title_token(tok) for tok in text.split())


# Nobiliary / surname particles. Title Case keeps them capitalized; we also
# emit a lowercase-particle form (van Gogh / de Gaulle / von Neumann, …).
_PARTICLES = frozenset(
    {
        "van",
        "von",
        "de",
        "der",
        "den",
        "des",
        "ter",
        "ten",
        "te",
        "zu",
        "zum",
        "zur",
        "da",
        "das",
        "dos",
        "do",
        "di",
        "du",
        "del",
        "della",
        "degli",
        "delle",
        "la",
        "le",
        "el",
        "y",
        "af",
        "av",
    }
)


def _family_particle_forms(family: str) -> list[str]:
    """Title-case family plus a form with known particles lowercased."""
    tokens = family.split()
    if len(tokens) < 2:
        return [family]
    lowered = [t.lower() if t.lower() in _PARTICLES else t for t in tokens]
    alt = " ".join(lowered)
    if alt == family:
        return [family]
    return [family, alt]


# Generational suffixes (Jr / Sr / II…). Not initials; expand common surfaces.
_GENERATIONAL_FORMS: dict[str, tuple[str, ...]] = {
    "jr": ("Jr.", "Jr", "Junior"),
    "junior": ("Jr.", "Jr", "Junior"),
    "sr": ("Sr.", "Sr", "Senior"),
    "senior": ("Sr.", "Sr", "Senior"),
    "ii": ("II",),
    "iii": ("III",),
    "iv": ("IV",),
}


def _generational_surfaces(tok: str) -> tuple[str, ...] | None:
    """Return Jr./Jr/Junior (etc.) surfaces, or None if tok is not generational."""
    key = tok.rstrip(",").lower().rstrip(".")
    return _GENERATIONAL_FORMS.get(key)


def peel_generational_suffix(text: str) -> tuple[str, tuple[str, ...] | None]:
    """Strip a trailing generational suffix; return (core, surfaces or None)."""
    tokens = text.split()
    if not tokens:
        return text, None
    surfaces = _generational_surfaces(tokens[-1])
    if surfaces is None:
        return text, None
    core = " ".join(tokens[:-1]).rstrip(",").strip()
    return core, surfaces


def _token_initial_forms(tok: str) -> list[str]:
    """Plausible initial spellings for one given-name token.

    Simple names → `E.`
    Hyphenated compounds (Jean-Michel) → `J.-M.`, `J-M`, `JM`
    """
    parts = [p for p in tok.split("-") if p]
    if not parts:
        return []
    if len(parts) == 1:
        return [f"{parts[0][0].upper()}."]
    letters = [p[0].upper() for p in parts]
    return [
        "-".join(f"{c}." for c in letters),  # J.-M.
        "-".join(letters),  # J-M
        "".join(letters),  # JM
    ]


def _space_hyphens(text: str) -> str:
    """Turn word hyphens into spaces, leaving initialisms intact."""
    out: list[str] = []
    for tok in text.split():
        core, comma = (tok[:-1], ",") if tok.endswith(",") else (tok, "")
        if _is_initialism_token(core):
            out.append(tok)
        else:
            out.append(core.replace("-", " ") + comma)
    return " ".join(out)


def latin_variants(given: str, family: str) -> list[str]:
    """Return plausible Western Latin-script spelling variants.

    Parameters
    ----------
    given:
        Given / first name(s). May contain multiple tokens
        (`Ernst August`, `José María`), hyphenated compounds
        (`Jean-Michel`), or initials (`E. A.`).
    family:
        Family / surname(s). May contain multiple tokens
        (`Lopez Fernandez`) or a hyphenated compound (`García-López`).

    Only forms commonly seen in Western documents, patents, and
    bibliographies are generated:

      word order    -- Given Family | Family, Given
      given-name    -- full | first only | all initials | first + rest-as-initials
                       (hyphenated givens also yield J.-M. / J-M / JM)
      case          -- Title Case | ALL CAPS  (no all-lowercase);
                       particles van/de/von/… also as lowercase
      diacritics    -- keep | strip (incl. ł→l, ø→o, æ→ae, …) |
                       German/Nordic digraph (ü→ue, ø→oe, ß→ss)
      compound sep  -- hyphen | space  (no glued `GarciaLopez`;
                       initialisms like J.-M. are not split)
      generational  -- trailing Jr/Sr/II… peeled then reattached as
                       Jr./Jr/Junior (etc.); never initialed

    Multi-standard Latin romanizations of non-Western names (Yurij/Yuri,
    Zhang/Chang, …) are not invented here — inject them via
    `latin_variants_with_alts` (see `generate_names_phase_2.py`).
    """
    given = given.strip()
    family = family.strip()
    if not family:
        return []

    # Peel Jr/Sr/II… from either side so given-reduction never initials them.
    family, fam_gen = peel_generational_suffix(family)
    given, given_gen = peel_generational_suffix(given)
    gen_surfaces = fam_gen or given_gen
    if not family:
        return []

    # Normalize to Title Case before generating structural variants so that
    # ALL-CAPS / mixed-case inputs do not leak into the output set.
    family = title_name(family)
    families = _family_particle_forms(family)
    given_spellings = [title_name(given)] if given else [""]

    def _all_inits_forms(words: list[str]) -> list[str]:
        form_lists = [_token_initial_forms(w) for w in words if w]
        if not form_lists:
            return []
        return [" ".join(combo) for combo in product(*form_lists)]

    def _compact_inits(words: list[str]) -> str:
        # Primary (first) initial form of each token, concatenated: E.A. / J.-M.P.
        return "".join(_token_initial_forms(w)[0] for w in words if w)

    def _first_plus_inits(words: list[str]) -> str | None:
        if len(words) < 2:
            return None
        # Skip tokens with no initials (e.g. bare "-" from "Jean - Michel").
        inits: list[str] = []
        for w in words[1:]:
            forms = _token_initial_forms(w)
            if forms:
                inits.append(forms[0])
        if not inits:
            return None
        return f"{words[0]} {' '.join(inits)}"

    def _structural_given_forms(givens: list[str]) -> list[str]:
        if not givens:
            return [""]
        candidates = [
            " ".join(givens),  # Ernst August / Jean-Michel
            givens[0],  # Ernst / Jean-Michel
        ]
        candidates.extend(_all_inits_forms(givens))  # E. A. / J.-M. / J-M / JM
        # Compact E.A. only for space-separated givens; hyphenated compounds
        # already contribute JM / J-M / J.-M. via _token_initial_forms.
        if len(givens) > 1 and all("-" not in g for g in givens):
            candidates.append(_compact_inits(givens))  # E.A.
        fp = _first_plus_inits(givens)
        if fp:
            candidates.append(fp)  # Ernst A. / Jean-Michel P.
        return list(dict.fromkeys(c for c in candidates if c))

    given_forms: list[str] = []
    seen_gf: set[str] = set()
    for spelling in given_spellings:
        for gf in _structural_given_forms(spelling.split() if spelling else []):
            if gf not in seen_gf:
                seen_gf.add(gf)
                given_forms.append(gf)
    if not given_forms:
        given_forms = [""]

    # Western word orders only: Given Family, and bibliographic Family, Given.
    base_forms: list[str] = []
    for fam in families:
        for gf in given_forms:
            if gf:
                base_forms.append(f"{gf} {fam}")
                base_forms.append(f"{fam}, {gf}")
            else:
                base_forms.append(fam)

    if gen_surfaces:
        base_forms = [f"{bf} {sfx}" for bf in base_forms for sfx in gen_surfaces]

    # Diacritic variants
    diac: list[str] = []
    for bf in base_forms:
        diac.append(bf)
        s = strip_diacritics(bf)
        if s != bf:
            diac.append(s)
        d = digraph(bf)
        if d != bf and d != s:
            diac.append(d)

    # Hyphenated name words → space-separated alternative (not glued);
    # leave initialisms such as J.-M. / J-M untouched.
    sep: list[str] = []
    for v in diac:
        sep.append(v)
        spaced = _space_hyphens(v)
        if spaced != v:
            sep.append(spaced)

    # Case: Title Case (already) + ALL CAPS. Skip all-lowercase.
    cased: list[str] = []
    for v in sep:
        cased.append(v)
        u = v.upper()
        if u != v:
            cased.append(u)

    seen: set[str] = set()
    out: list[str] = []
    for v in cased:
        v = v.strip()
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def normalize_string_list(value: Any, max_n: int) -> list[str]:
    """Coerce a tool arg to a capped, deduped list of non-empty strings."""
    if value is None or value == "":
        return []
    if isinstance(value, list):
        items: list[Any] = value
    elif isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if s.startswith("["):
            try:
                parsed = json.loads(s)
            except json.JSONDecodeError:
                parsed = None
            items = parsed if isinstance(parsed, list) else [s]
        else:
            items = [s]
    else:
        items = [value]
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= max_n:
            break
    return out


def split_display_name(name: str) -> tuple[str, str]:
    """Best-effort given/family split for an alt Latin display string."""
    name = name.strip()
    if not name:
        return "", ""
    # Trailing Jr/Sr/II… is not a surname; peel, split, reattach to family.
    name, gen_surfaces = peel_generational_suffix(name)
    name = name.rstrip(",").strip()
    if not name:
        return "", ""
    if "," in name:
        left, right = name.split(",", 1)
        given, family = right.strip(), left.strip()
    else:
        parts = name.split()
        if len(parts) < 2:
            given, family = "", name
        else:
            given, family = " ".join(parts[:-1]), parts[-1]
    if gen_surfaces and family:
        family = f"{family} {gen_surfaces[0]}"
    return given, family


def latin_variants_with_alts(
    given: str,
    family: str,
    alt_latin: list[str],
    *,
    base: list[str] | None = None,
) -> list[str]:
    """Primary `latin_variants` plus capped alt romanizations expanded the same way.

    Each alt is a full display string (`Given Family` or `Family, Given`).
    Drops alts whose family token is one of the primary given tokens (typical
    given/family swap pollution, e.g. `Yi Sedol` for primary `Sedol Lee`).

    `base`, when given, replaces the freshly computed `latin_variants(given,
    family)` as the starting variant list — used by phase 2 to merge onto the
    `la{}` forms already produced by phase 1 instead of recomputing them.
    """
    out = list(base) if base is not None else latin_variants(given, family)
    seen = set(out)
    primary_given_toks = {t.lower() for t in given.split() if t}
    for alt in normalize_string_list(alt_latin, MAX_ALT_LATIN):
        g, f = split_display_name(alt)
        if not f:
            continue
        if primary_given_toks and f.lower() in primary_given_toks:
            continue
        for v in latin_variants(g, f):
            if v not in seen:
                seen.add(v)
                out.append(v)
    return out


def given_family_from_la_forms(la_forms: list[str]) -> tuple[str, str]:
    """Recover (given, family) from a phase-1 `la{}` variant list.

    Prefers a comma-form entry (`Family, Given`): family may be multi-token
    (`Lopez Fernandez, Ivan`), and a naive last-token split of the `Given
    Family` form would misplace the boundary. Falls back to the first la form
    (naive split) when no comma form is present, e.g. family-only names.
    """
    for val in la_forms:
        if "," in val:
            return split_display_name(val)
    if la_forms:
        return split_display_name(la_forms[0])
    return "", ""


# ---------------------------------------------------------------------------
# Cluster-corpus line format (phase 1 output / phase 2 input+output)
# ---------------------------------------------------------------------------

_CLUSTER_CELL_RE = re.compile(r"^([a-z]{2})\{(.*)\}$")


def parse_cluster_line(line: str) -> list[tuple[str, str]] | None:
    """Parse a tab-separated `tag{value}` cluster line into ordered pairs.

    Returns None if any cell doesn't match the `tag{value}` shape, so callers
    can pass the line through unmodified rather than mangling something that
    isn't actually cluster-corpus format.
    """
    cells = line.split("\t")
    parsed: list[tuple[str, str]] = []
    for cell in cells:
        m = _CLUSTER_CELL_RE.match(cell)
        if not m:
            return None
        parsed.append((m.group(1), m.group(2)))
    return parsed


def format_cluster_line(cells: list[tuple[str, str]]) -> str:
    """Inverse of `parse_cluster_line`."""
    return "\t".join(f"{tag}{{{val}}}" for tag, val in cells)


# ---------------------------------------------------------------------------
# Agent tools (name extraction)
# ---------------------------------------------------------------------------

EXTRACT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "extract_name",
            "description": (
                "DEFAULT tool — call whenever ≥2 recognisable name tokens are present.\n"
                "Handles every ordering convention:\n"
                "  Western    : Ernst A. Mayr          → given='Ernst A.', family='Mayr'\n"
                "  ALL-CAPS   : COBB JAMES S.           → given='James S.', family='Cobb'\n"
                "  Comma split: LOPEZ FERNANDEZ, IVAN   → given='Ivan', family='Lopez Fernandez'\n"
                "  Semicolon  : NATOUR; GHALEB           → given='Ghaleb', family='Natour'\n"
                "  Lead init  : J SPIELER KARL           → given='Karl J.', family='Spieler'\n"
                "Lines may contain trailing noise (addresses, IDs) — strip it, extract only the name.\n"
                "When unsure whether to extract or reject: ALWAYS call extract_name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "given": {
                        "type": "string",
                        "description": (
                            "Given name(s): first name plus any middle name(s) or initial(s).\n"
                            "Format: Title-case full words; initials as single uppercase letter + period.\n"
                            "MUST contain at least one full word — a bare initial alone is NOT valid.\n"
                            "Examples: 'Ernst A.', 'James S.', 'José María', "
                            "'Charlotte C.', 'Thomas P.', 'Karl J.', 'Daoxi', 'Ivan'"
                        ),
                    },
                    "family": {
                        "type": "string",
                        "description": (
                            "Family / surname(s).\n"
                            "Format: Title-case; preserve hyphens and spaces exactly as written.\n"
                            "MUST be a full word — a bare initial alone is NOT valid.\n"
                            "Examples: 'Mayr', 'Cobb', 'Tan', 'Guida', "
                            "'García-López', 'Lopez Fernandez', 'Natour', 'Spieler'"
                        ),
                    },
                },
                "required": ["given", "family"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "not_a_person_name",
            "description": (
                "Call ONLY when the input CANNOT yield a valid given+family pair.\n"
                "Valid rejection cases (high bar — reject very little):\n"
                "  • Single token, no split possible : 'SCHNECK', 'Madonna'\n"
                "  • Initial(s) only, no full word   : 'Filion G', 'G. Filion', 'I.L. SMITH'\n"
                "  • Org / place / product / junk    : 'Acme Corp', 'Boston MA'\n"
                "DO NOT reject — extract these instead:\n"
                "  • ALL-CAPS multi-token lines : 'COBB JAMES S.', 'KAUFMAN JAN',\n"
                "                                 'ALLISON CHARLOTTE C', 'TAN DAOXI'\n"
                "  • Lead initial + ≥2 full words: 'J SPIELER KARL'\n"
                "  • Name with trailing noise    : 'John S. Doe, 12 Main St, Boston MA'\n"
                "Default: when in doubt, call extract_name instead of this tool."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]

EXTRACT_SYSTEM_PROMPT = (
    "You are a person-name extractor for a name-variant pipeline. Call exactly one tool per input. No free text."
)

_EXTRACT_USER_CONTRACT = """\
Extract the person name from the line below. Call extract_name(given, family) \
when a name with both given and family parts is present. Call \
not_a_person_name() only for clearly unusable input.

### Order heuristics
- Comma or semicolon → "Family, Given": given = after separator, family = before.
- Last token is a full word → Western: last = family, rest = given.
- ALL-CAPS patent "FAMILY GIVEN" or trailing initial → first = family, rest = given.
- Leading bare initial + FAMILY GIVEN (e.g. "J SPIELER KARL") → family = middle \
full word, given = last full word + the initial (Karl J.).

### Rules
- Do not invent names absent from the line.
- Ignore addresses / IDs — extract only the name.
- Titles / honorifics are NOT name parts — never put one in given or family: \
Dr/Prof/Ing/Arch/Dipl/Mr/Mrs/Ms/Rev/Fr/Sir/Dame/Col/Hon/PhD/MD (English/German/\
Spanish), Mme/Mlle (French), Dott/Avv (Italian), and equivalents from any \
language.
- Fix obvious OCR errors (5→S, 0→O, 1→l/I). Do not "correct" plausible spellings.
- Prefer extract_name whenever ≥2 full-word tokens look like a person name.
- A bare initial among ≥2 full words is fine (middle/leading); do not reject for that alone.
- Reject only: orgs/places/junk, mononyms, initial-only forms (no full given+family pair). \
Do not invent a given from an initial.
- When unsure between extract and reject, call extract_name.

### Examples
"COBB JAMES S." → given="James S.", family="Cobb"
"GUIDA THOMAS P" → given="Thomas P.", family="Guida"
"ALLISON CHARLOTTE C" → given="Charlotte C.", family="Allison"
"KAUFMAN JAN" → given="Jan", family="Kaufman"
"TAN DAOXI" → given="Daoxi", family="Tan"
"J SPIELER KARL" → given="Karl J.", family="Spieler"
"Ernst A. Mayr" → given="Ernst A.", family="Mayr"
"Peter Dr. Flury" → given="Peter", family="Flury" (title is not a name part)
"Edmund Prof. Dr. Wax" → given="Edmund", family="Wax"
"LOPEZ FERNANDEZ, IVAN" → given="Ivan", family="Lopez Fernandez"
"NATOUR; GHALEB" → given="Ghaleb", family="Natour"
"John S. Doe, 12 Main St, Boston MA" → given="John S.", family="Doe"
"John 5. Doe" → given="John S.", family="Doe" (OCR: 5→S)
Reject: "Filion G" / "G. Filion" / "I.L. SMITH" / "SCHNECK" / "Madonna" / "Acme Corp"
Do NOT reject ALL-CAPS two-or-more-token inventor lines — extract them.

### Input
"""


def make_extract_user_message(line: str) -> str:
    return f"{_EXTRACT_USER_CONTRACT}{line}"


def make_extract_retry_user_message(line: str) -> str:
    return f"{_EXTRACT_USER_CONTRACT}{line}\n\nYou MUST respond with a tool call. No free text."


def not_a_person_name() -> None:
    """Marker tool: the input line does not contain a person name."""
    return None


# ---------------------------------------------------------------------------
# vLLM setup
# ---------------------------------------------------------------------------


def build_llm() -> LLM:
    return LLM(
        model=MODEL_NAME,
        tensor_parallel_size=2,
        quantization="fp8",
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        enable_prefix_caching=True,
    )


def make_extract_sampling_params() -> SamplingParams:
    # Structured tool calls: greedy, short completion.
    return SamplingParams(
        max_tokens=MAX_EXTRACT_TOKENS,
        temperature=0.0,
        top_p=1.0,
        stop=["</tool_call>"],
        include_stop_str_in_output=True,
    )


def make_extract_retry_sampling_params() -> SamplingParams:
    return SamplingParams(
        max_tokens=MAX_EXTRACT_TOKENS,
        temperature=0.3,
        top_p=0.95,
        stop=["</tool_call>"],
        include_stop_str_in_output=True,
    )


# ---------------------------------------------------------------------------
# LLM output parsers
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
# Qwen3 XML tool-call format (offline LLM.generate does not apply serve parsers).
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^>\s]+)\s*>(.*?)</function>\s*</tool_call>",
    flags=re.DOTALL,
)
_PARAM_RE = re.compile(r"<parameter=([^>\s]+)\s*>(.*?)</parameter>", flags=re.DOTALL)


def strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text)


def coerce_param_value(raw: str) -> Any:
    """Strip a parameter body; parse JSON arrays/objects when present."""
    s = raw.strip()
    if s.startswith("[") or s.startswith("{"):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return s
    return s


def parse_tool_calls(text: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse tool calls from raw model text.

    Supports Qwen3 XML `<tool_call><function=...>...</function></tool_call>`
    and a JSON fallback `[{"name": ..., "arguments": {...}}, ...]`.
    """
    text = strip_thinking(text)
    calls: list[tuple[str, dict[str, Any]]] = []

    for name, body in _TOOL_CALL_RE.findall(text):
        args: dict[str, Any] = {k: coerce_param_value(v) for k, v in _PARAM_RE.findall(body)}
        if not args:
            body_stripped = body.strip()
            if body_stripped.startswith("{"):
                try:
                    parsed = json.loads(body_stripped)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    args = parsed
        calls.append((name.strip(), args))

    if calls:
        return calls

    # JSON fallback (some chat templates / demos emit a bare list).
    stripped = text.strip()
    if stripped.startswith("["):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return []
        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue
                name = item.get("name")
                args = item.get("arguments", item.get("parameters", {}))
                if isinstance(name, str) and isinstance(args, dict):
                    calls.append((name, args))
    return calls


def heal_given_family(given: str, family: str) -> tuple[str, str]:
    """If family is empty, take the last non-initial token of given as family.

    Repairs the common tool-call mistake of stuffing the whole name into
    `given` with `family=""`. Bare trailing initials (`R.`, `P.`) are not
    treated as surnames.
    """
    given, family = given.strip(), family.strip()
    if family or not given:
        return given, family
    parts = given.split()
    if len(parts) < 2:
        return given, family
    last = parts[-1]
    core = last.rstrip(".")
    if len(core) == 1 and core.isalpha():
        return given, family
    return " ".join(parts[:-1]), last


def dispatch_extract_tool_call(name: str, args: dict[str, Any]) -> tuple[str, list[str] | None] | None:
    """Execute one extraction tool call.

    Returns
    -------
    `("extract_name", variants)`
        Person name accepted; `variants` is the Latin spelling list.
    `("not_a_person_name", None)`
        Line rejected as non-name.
    `("incomplete_name", None)`
        Line had a person-ish string but failed the full given+family gate.
    `None`
        Unknown / malformed tool call.
    """
    if name == "not_a_person_name":
        not_a_person_name()
        return "not_a_person_name", None
    if name == "extract_name":
        given = str(args.get("given", "")).strip()
        family = str(args.get("family", "")).strip()
        given, family = heal_given_family(given, family)
        args["given"], args["family"] = given, family
        if not is_usable_person_name(given, family):
            return "incomplete_name", None
        return "extract_name", latin_variants(given, family)
    return None


# ---------------------------------------------------------------------------
# I/O + chat helpers
# ---------------------------------------------------------------------------


def read_lines(path: str) -> Iterator[str]:
    """Yield non-empty stripped lines from a gzipped or plain-text file."""
    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield line


def display_name(given: str, family: str) -> str:
    given, family = given.strip(), family.strip()
    return f"{given} {family}".strip() if given else family


def apply_chat(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    tools: list[dict[str, Any]] | None = None,
    enable_thinking: bool = False,
) -> list[int]:
    """Build a chat prompt once as token ids (no second tokenize in generate)."""
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "return_dict": False,  # transformers v5 defaults True → BatchEncoding
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    if tools is not None:
        kwargs["tools"] = tools
    try:
        ids = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        # Older tokenizers reject return_dict.
        kwargs.pop("return_dict")
        ids = tokenizer.apply_chat_template(messages, **kwargs)
    # Normalize BatchEncoding / batched / string fallbacks to flat list[int].
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    elif isinstance(ids, dict) and "input_ids" in ids:
        ids = ids["input_ids"]
    if isinstance(ids, str):
        ids = tokenizer.encode(ids, add_special_tokens=False)
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(t) for t in ids]


def prompt_fits(prompt_ids: list[int], max_new_tokens: int) -> bool:
    return len(prompt_ids) + max_new_tokens <= MAX_MODEL_LEN


def tokens_prompt(prompt_ids: list[int]) -> TokensPrompt:
    return TokensPrompt(prompt_token_ids=prompt_ids)
