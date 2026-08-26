"""Add alternate Latin romanizations to phase-1 cluster lines.

Sister pipeline to `generate_names_phase_1.py`. Phase 1 extracts a name,
expands deterministic Western Latin surfaces (order, initials, case,
diacritics), and transliterates it into eleven other scripts. Phase 2 reads
phase 1's cluster output, asks the LLM for plausible multi-standard Latin
romanizations of the *same* person (Viktor→Victor, Zhang→Chang, …), and
merges them into the `la{}` side of the same cluster via
`latin_variants_with_alts`.

Pipeline:
  Parse cluster  -- split each `la{...}\\tcy{...}\\t...` line into ordered
                    (tag, value) cells; recover (given, family) from the
                    `la{}` forms already produced by phase 1
                    (`given_family_from_la_forms`)
  Alt Latin      -- Qwen3 tool call: `emit_alt_latin(alts)`
                    (empty for ordinary Western names; one retry if missing)
  Merge          -- `latin_variants_with_alts(given, family, alts,
                    base=la_forms)` expands each alt the same way phase 1
                    expands the primary name, and appends only new forms

Input : gzipped or plain-text file of phase-1 cluster lines (tab-separated
        `tag{value}` cells: one or more `la{...}` plus one or more of each of
        the eleven non-Latin script tags).
Output: the same cluster lines, `la{}` cells augmented for names judged
        non-Western (their family name reads Slavic/CJK/Arabic/Indic/etc.)
        and left byte-identical for ordinary Western names. Non-Latin cells
        are always passed through unchanged. A line that doesn't parse as
        cluster-corpus format is passed through unmodified rather than
        dropped.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# Allow `python runners/generate_names_phase_2.py` to import the shared core.
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

MAX_ALT_TOKENS = 192

# ---------------------------------------------------------------------------
# Tools / prompts (alt-Latin pass)
# ---------------------------------------------------------------------------

ALT_LATIN_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "emit_alt_latin",
            "description": (
                "Emit up to 3 common alternate Latin romanizations of the SAME "
                "person as full 'Given Family' (or 'Family, Given') strings. "
                "Use only for clearly non-Western names with well-known "
                "multi-standard spellings across countries/languages "
                "(e.g. Yurij→Yuri/Youri, Zhang→Chang, Lee→Yi). "
                "Pass an empty list for ordinary Western names."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "alts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Up to 3 alternate Latin display names. Each entry "
                            "must be a full 'Given Family' or 'Family, Given' "
                            "string for the SAME person — different romanization "
                            "only, not a different person, nickname, or "
                            "translation. Empty list when no well-known fork exists."
                        ),
                    },
                },
                "required": ["alts"],
            },
        },
    },
]

ALT_LATIN_SYSTEM_PROMPT = (
    "You are a Latin-romanization specialist for inventor / bibliographic names. "
    "Call exactly one tool: emit_alt_latin(alts). No free text."
)

_ALT_LATIN_USER_CONTRACT = """\
Given a person name in Latin script, call emit_alt_latin(alts) with up to 3 \
common alternate Latin romanizations of the SAME person.

### First, judge the name's origin from the FAMILY name, not the given name
A given name can look foreign yet belong to a Western person (a Hungarian or \
German "Viktor", a French "Marc"), so decide non-Western origin from the family \
name's shape: unambiguous Slavic suffixes (-ov/-ev/-ova/-eva/-off/-sky/-enko), \
Chinese/Korean/Japanese romanized syllables, Arabic/Persian patterns \
(al-/bin/ibn prefixes), Indic patterns, etc. Skip suffixes that also occur in \
ordinary Western surnames (-in as in Martin, -ich as in Aldrich, -i as in \
Rossi) — those alone are not evidence of non-Western origin. Only once the \
family name reads as non-Western does a fork on the given name's spelling apply.
- Family name reads non-Western (e.g. ends in -ov/-ev/-sky/-enko, or is a \
  Chinese/Korean/Arabic/Indic surname): also fork well-known cross-language \
  given-name spellings for that family's language, even if the given name \
  alone looks plausibly Western (Viktor→Victor, Yuriy/Jurij→Yuri/Youri, \
  Aleksandr→Alexander, Mikhail→Michael, Nikolai→Nicholas).
- Family name reads ordinary Western European / Anglo (Dupont, Mayr, Filion, \
  García, Smith, Bauer, Rossi): pass alts=[] even if the given name has a \
  foreign-language cognate — do not romanize a Western surname's bearer as if \
  they were from elsewhere (a "Victor Dupont" is not plausibly "Viktor Dupont").

### When to emit alts
- Non-Western names whose Latin spelling varies by country or standard:
  Slavic (Yurij/Yuri/Youri/Jurij), Chinese (Zhang/Chang, Wei/Wai), Korean \
  (Lee/Yi/Rhee), Japanese (Sato/Satou), Arabic (Mohammed/Muhammad/Mohamed), \
  Greek (Giorgos/Yorgos), Indic (Ravi/Raavi), etc.
- Prefer forms a patent clerk or librarian in France, the UK, Germany, or the \
  US might actually write.

### When to pass alts=[]
- Ordinary Western European / Anglo names with a stable Latin spelling \
  (Ernst Mayr, Guillaume Filion, José García), even when a given name in \
  isolation also exists in another language.
- Do not invent phonetic near-misses, nicknames, or translations.
- Do not emit order/initials/case/diacritic variants — Python expands those.

### Format
- Each alt is a full display string: "Given Family" (preferred) or "Family, Given".
- Keep middle initials when present in the input (Yurij P. Kirin → Yuri P. Kirin).
- Do not repeat the input spelling itself.

### Examples
"Yurij P. Kirin" → alts=["Yuri P. Kirin", "Youri P. Kirin"]
"Wei Zhang" → alts=["Wei Chang"]
"Sedol Lee" → alts=["Sedol Yi", "Sedol Rhee"]
"Viktor T. Skokov" → alts=["Victor T. Skokov"]  # -ov surname is Slavic; fork the given name too
"Ernst A. Mayr" → alts=[]
"Guillaume Filion" → alts=[]
"Victor Dupont" → alts=[]  # Dupont is a Western surname; do not offer "Viktor Dupont"

### Input
"""


def _make_alt_latin_user_message(name: str) -> str:
    return f"{_ALT_LATIN_USER_CONTRACT}{name}"


def _make_alt_latin_retry_user_message(name: str) -> str:
    return (
        f"{_ALT_LATIN_USER_CONTRACT}{name}\n\nYou MUST respond with a tool call emit_alt_latin(alts=...). No free text."
    )


# ---------------------------------------------------------------------------
# Sampling / dispatch
# ---------------------------------------------------------------------------


def _make_alt_latin_sampling_params():
    return core.SamplingParams(
        max_tokens=MAX_ALT_TOKENS,
        temperature=0.2,
        top_p=0.95,
        stop=["</tool_call>"],
        include_stop_str_in_output=True,
    )


def _make_alt_latin_retry_sampling_params():
    return core.SamplingParams(
        max_tokens=MAX_ALT_TOKENS,
        temperature=0.5,
        top_p=0.95,
        stop=["</tool_call>"],
        include_stop_str_in_output=True,
    )


def _parse_emit_alt_latin(raw: str) -> list[str] | None:
    """Parse emit_alt_latin tool call; None if missing or wrong tool."""
    calls = core.parse_tool_calls(raw)
    if not calls:
        return None
    name, args = calls[0]
    if name != "emit_alt_latin":
        return None
    return core.normalize_string_list(args.get("alts", []), core.MAX_ALT_LATIN)


# ---------------------------------------------------------------------------
# Cluster line handling
# ---------------------------------------------------------------------------


def _split_cluster(
    cells: list[tuple[str, str]],
) -> tuple[list[str], list[tuple[str, str]]]:
    """Split parsed cluster cells into (la forms, other cells, order preserved)."""
    la_forms = [v for t, v in cells if t == "la"]
    other_cells = [(t, v) for t, v in cells if t != "la"]
    return la_forms, other_cells


def _rebuild_cluster(la_forms: list[str], other_cells: list[tuple[str, str]]) -> str:
    return core.format_cluster_line([("la", v) for v in la_forms] + other_cells)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Add alternate Latin romanizations to phase-1 cluster lines.")
    parser.add_argument(
        "input",
        help="Input text (.gz or plain): phase-1 cluster output, one cluster per line.",
    )
    parser.add_argument("--output", "-o", default="-", help="Output file (default: stdout).")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=core.MAX_NUM_SEQS,
        help="LLM batch size (default: %(default)s).",
    )
    args = parser.parse_args(argv)

    lines = list(core.read_lines(args.input))
    log.info(f"Loaded {len(lines)} lines from {args.input}")

    llm = core.build_llm()
    tokenizer = llm.get_tokenizer()
    alt_params = _make_alt_latin_sampling_params()
    alt_retry_params = _make_alt_latin_retry_sampling_params()

    out_fh = open(args.output, "w", encoding="utf-8") if args.output != "-" else sys.stdout
    try:
        for batch_start in range(0, len(lines), args.batch_size):
            batch = lines[batch_start : batch_start + args.batch_size]

            # batch_out[i] holds the final output line for batch[i]; filled
            # immediately for anything passed through unmodified, filled
            # after the LLM round(s) for the rest. Every line in `batch`
            # produces exactly one output line.
            batch_out: list[str | None] = [None] * len(batch)
            # (batch_idx, given, family, label, la_forms, other_cells, prompt_ids)
            llm_items: list[tuple[int, str, str, str, list[str], list[tuple[str, str]], list[int]]] = []

            for i, line in enumerate(batch):
                cells = core.parse_cluster_line(line)
                if cells is None:
                    log.warning(f"Unparseable cluster line, passing through unmodified: {line!r:.80}")
                    batch_out[i] = line
                    continue
                la_forms, other_cells = _split_cluster(cells)
                if not la_forms:
                    log.warning(f"No la{{}} forms in cluster line, passing through unmodified: {line!r:.80}")
                    batch_out[i] = line
                    continue
                given, family = core.given_family_from_la_forms(la_forms)
                if not family:
                    log.warning(f"Could not recover family name, passing through unmodified: {line!r:.80}")
                    batch_out[i] = line
                    continue
                label = core.display_name(given, family)
                prompt_ids = core.apply_chat(
                    tokenizer,
                    [
                        {"role": "system", "content": ALT_LATIN_SYSTEM_PROMPT},
                        {"role": "user", "content": _make_alt_latin_user_message(label)},
                    ],
                    tools=ALT_LATIN_TOOLS,
                    enable_thinking=False,
                )
                if not core.prompt_fits(prompt_ids, MAX_ALT_TOKENS):
                    log.warning(f"Overlong alt-latin prompt ({len(prompt_ids)} tokens), passing through: {label!r:.80}")
                    batch_out[i] = line
                    continue
                llm_items.append((i, given, family, label, la_forms, other_cells, prompt_ids))

            resolved: dict[int, list[str]] = {}
            if llm_items:
                alt_outputs = llm.generate(
                    [core.tokens_prompt(ids) for *_r, ids in llm_items],
                    alt_params,
                    use_tqdm=False,
                )
                retry_items: list[tuple[int, str, str, str, list[str], list[tuple[str, str]]]] = []
                for (i, given, family, label, la_forms, other_cells, _ids), output in zip(llm_items, alt_outputs):
                    alts = _parse_emit_alt_latin(output.outputs[0].text)
                    if alts is None:
                        log.warning(f"No emit_alt_latin call for '{label}'; retrying")
                        retry_items.append((i, given, family, label, la_forms, other_cells))
                        continue
                    resolved[i] = alts

                if retry_items:
                    retry_batch: list[tuple[int, str, str, str, list[str], list[tuple[str, str]], list[int]]] = []
                    for i, given, family, label, la_forms, other_cells in retry_items:
                        prompt_ids = core.apply_chat(
                            tokenizer,
                            [
                                {"role": "system", "content": ALT_LATIN_SYSTEM_PROMPT},
                                {"role": "user", "content": _make_alt_latin_retry_user_message(label)},
                            ],
                            tools=ALT_LATIN_TOOLS,
                            enable_thinking=False,
                        )
                        if core.prompt_fits(prompt_ids, MAX_ALT_TOKENS):
                            retry_batch.append((i, given, family, label, la_forms, other_cells, prompt_ids))
                        else:
                            resolved[i] = []
                    if retry_batch:
                        retry_outputs = llm.generate(
                            [core.tokens_prompt(ids) for *_r, ids in retry_batch],
                            alt_retry_params,
                            use_tqdm=False,
                        )
                        for (i, given, family, label, la_forms, other_cells, _ids), output in zip(
                            retry_batch, retry_outputs
                        ):
                            alts = _parse_emit_alt_latin(output.outputs[0].text)
                            if alts is None:
                                log.warning(f"No emit_alt_latin on retry for '{label}'; leaving unmodified")
                                alts = []
                            resolved[i] = alts

                augmented = 0
                for i, given, family, label, la_forms, other_cells, _ids in llm_items:
                    alts = resolved.get(i, [])
                    merged_la = core.latin_variants_with_alts(given, family, alts, base=la_forms)
                    batch_out[i] = _rebuild_cluster(merged_la, other_cells)
                    if alts:
                        log.info(f"Alt Latin for '{label}': {alts}")
                        augmented += 1

            for i, out_line in enumerate(batch_out):
                if out_line is None:
                    # Should not happen (every path above fills its slot), but
                    # the contract is "never destroy phase 1's work" — fall
                    # back to the original line rather than raising mid-run.
                    log.warning(f"Unfilled output slot for line {batch[i]!r:.80}; passing through unmodified")
                    out_line = batch[i]
                print(out_line, file=out_fh)

            log.info(
                f"Processed {min(batch_start + args.batch_size, len(lines))}/{len(lines)} "
                f"({len(llm_items)} sent to LLM, {augmented if llm_items else 0} augmented)"
            )
    finally:
        if out_fh is not sys.stdout:
            out_fh.close()


if __name__ == "__main__":
    main()
