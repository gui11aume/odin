# Cross-script name variant generation

Shared helpers live in `generate_names_core.py`. Phase scripts import from there:

- `generate_names_phase_1.py` — extract → Latin variants → cross-script `emit_scripts` → post-process
- `generate_names_phase_2.py` — reads phase 1's cluster output → alt-Latin romanizations → `latin_variants_with_alts`

Phase 1 builds training clusters of person-name spellings across Latin and eleven other writing systems. Each accepted input becomes one tab-separated cluster of `tag{form}` tokens. Incomplete clusters (any of the eleven non-Latin scripts missing after per-tag mini-retry) are dropped.

Phase 2 is a second pass **over phase 1's output**, not a second run over the raw input: it reads the same cluster lines phase 1 wrote, recovers the underlying (given, family) from the `la{}` forms already present, and asks the LLM whether the person's family name reads as non-Western — Slavic, CJK, Arabic, Indic, etc. — and if so, which alternate Latin romanizations of the given name a Western record might also use (Viktor→Victor, Zhang→Chang, …). It folds any such alternates into the `la{}` side of the *same* cluster and passes every other cell through unchanged. Ordinary Western names come out byte-identical to their phase-1 line.

The pipeline is intentionally hybrid: the LLM decides *what* the name is and *how* it should look in other scripts; deterministic Python expands Latin surface forms and applies a few script-specific post-process rules.

---

## Phase 1: input and output

**Input.** A plain-text or gzipped file, one candidate string per line. Lines are free-form: they may be a clean name, a bibliographic “Family, Given” string, or a name mixed with address, affiliation, or OCR noise. Non-name lines (organizations, places, products, junk) are rejected.

**Output.** One line per inventor in cluster-corpus format. Every complete cluster has:

- one or more Latin forms tagged `la{...}`
- at least one form for each of: `cy`, `gk`, `ab`, `cn`, `jp`, `kr`, `dv`, `hb`, `th`, `gg`, `am`

Example shape:

```text
la{Ernst A. Mayr}	la{ERNST MAYR}	…	cy{Эрнст Майр}	cn{…}
```

Only complete clusters are written. Partial transliterations after retry are discarded rather than emitted half-filled.

---

## Phase 1: pipeline overview

Four stages run in batches via vLLM (`Qwen/Qwen3.6-35B-A3B-FP8`, thinking off):

1. **Name extraction** — tool call: `extract_name(given, family)` or `not_a_person_name()`. Greedy decoding; one retry with pointed feedback if no tool call appears.
2. **Latin variants** — pure Python from the extracted given/family (see below).
3. **Cross-script transliteration** — same model, low temperature; three `emit_scripts(...)` passes over script groups (A: `cy`/`gk`/`hb`/`ab`; B: `cn`/`jp`/`kr`; C: `dv`/`th`/`gg`/`am`), one seed string per tag. Incomplete → one mini-retry per missing tag (higher temperature, narrow tool), then drop.
4. **Post-process** — OpenCC Simplified↔Traditional for Chinese; short CJK word-order flips for `cn`/`kr`; Cyrillic FIO abbreviation/order expansion for `cy`; Greek/Hebrew given–family abbreviation/order expansion for `gk`/`hb`; drop few-shot copy-paste contaminants; warn on Unicode-block mismatches (do not silently rewrite).

---

## Phase 1, stage 1: extraction agent

The extraction agent’s only job is to decide whether a person name is present and, if so, to split it into given and family parts. It must call **exactly one** tool and emit no free text.

| Tool | When |
|------|------|
| `extract_name(given, family)` | Line has a person name with both given and family (full-word tokens; middle initials OK). Prefer this when unsure. |
| `not_a_person_name()` | Clearly unusable: org/place/junk, mononym, or initial-only incomplete (`Filion G`, `I.L. SMITH`). Not for ALL-CAPS multi-token inventor lines. |

**Intentions for the agent:**

- Extract the name when it is embedded with other fields; do not put addresses or IDs into `given` / `family`.
- Fix obvious OCR/transcription errors in the name itself (`5`→`S`, `0`→`O`, `1`→`l`/`I` when clearly a letter). Do not “correct” plausible spellings.
- Order: comma or semicolon `Family, Given` → split on separator; last token a full word → Western (last = family); ALL-CAPS / patent `FAMILY GIVEN M.` or trailing initial → first = family, rest = given; leading bare initial + `FAMILY GIVEN` (e.g. `J SPIELER KARL`) → family = middle full word, given = last + initial.
- Multi-token givens and hyphenated families are fine. Middle/leading initials OK when ≥2 full-word tokens remain. Reject only mononyms and clearly initial-only forms (`Filion G`, `G. Filion`, `I.L. SMITH`, `Madonna`). When unsure, extract.
- Do not invent names that are not in the line (do not expand `G` into a guessed given name).

Illustrative mappings the prompt encodes:

| Input fragment | given | family |
|----------------|-------|--------|
| `José María García-López` | `José María` | `García-López` |
| `LOPEZ FERNANDEZ, IVAN` | `Ivan` | `Lopez Fernandez` |
| `NATOUR; GHALEB` | `Ghaleb` | `Natour` |
| `COBB JAMES S.` | `James S.` | `Cobb` |
| `ALLISON CHARLOTTE C` | `Charlotte C.` | `Allison` |
| `KAUFMAN JAN` | `Jan` | `Kaufman` |
| `LOEWE CHARLES` | `Charles` | `Loewe` |
| `TAN DAOXI` | `Daoxi` | `Tan` |
| `J SPIELER KARL` | `Karl J.` | `Spieler` |
| `Ernst A. Mayr` | `Ernst A.` | `Mayr` |
| `Guillaume Filion` | `Guillaume` | `Filion` |
| `John S. Doe, 12 Main St, …` | `John S.` | `Doe` |
| `Filion G` / `SCHNECK` / `Madonna` | → `not_a_person_name()` | |

---

## Latin variants (deterministic)

Once given and family are known, Python generates Western-document Latin spellings only—forms commonly seen in patents, bibliographies, and catalogs. The expansions cover:

- **Word order:** `Given Family` and `Family, Given` (no Eastern family-first Latin reorder).
- **Given-name reduction:** full given(s); first token only; all initials; first + rest as initials. Hyphenated givens also yield `J.-M.` / `J-M` / `JM`.
- **Case:** Title Case and ALL CAPS (no all-lowercase). Multi-token families also emit a form with known particles lowercased (`van`/`von`/`de`/`di`/`da`/…), e.g. `Ludwig Van Beethoven` and `Ludwig van Beethoven`.
- **Diacritics:** keep; strip (NFD marks plus non-decomposing letters: `ł`→`l`, `ø`→`o`, `æ`→`ae`, `ð`→`d`, `þ`→`th`, `đ`→`d`, `ħ`→`h`, `œ`→`oe`, `ı`→`i`, `ĳ`→`ij`); German/Nordic digraph (`ü`→`ue`, `ø`→`oe`, `ß`→`ss`, …).
- **Compound separators:** hyphen or space (not glued `GarciaLopez`). Initialisms like `J.-M.` are left intact.
- **Korean surname romanizations** (Latin-field aliases, not Hangul): `Lee`/`Li`/`Yi`/`Rhee`/`Ree`/`Ly`; `Park`/`Pak`/`Bak`; `Choi`/`Choe`/`Chey`; `Kim`. Hangul itself is left to the `kr{}` pass and is treated as stable.
- **Arabic `Al-`/`El-` prefixes** (again Latin-field): when a prefix is already present, expand to hyphen/space/glued and dropped-prefix forms (`Al-Hassan`, `El Hassan`, `Alhassan`, `Hassan`, …). Optional patronymic or tribal segments that may appear or vanish in databases are **not** expanded.

These Latin forms become the `la{...}` side of the cluster and also supply the display label used for transliteration (`Given Family`). Most patent-database variation for Korean and Arabic names lives here, not under `kr{}` or `ab{}`.

---

## Phase 1, stage 3: transliteration agent

The transliteration agent receives a single Latin display name and must call **exactly one** tool—`emit_scripts(...)`—per pass, with one seed string per script parameter in that pass. No free text and no extra spelling variants; post-processing owns multiplicity for `cy`/`gk`/`hb`/`cn`/`kr`.

First pass is three script-group calls (same model; tools/prompts list only that group's tags):

| Pass | Tags |
|------|------|
| A | `cy`, `gk`, `hb`, `ab` (+ `cy_alt` / `ab_alt` when useful) |
| B | `cn`, `jp`, `kr` (+ `jp_kana`, `cn_alt` / `jp_alt` / `kr_alt` when useful) |
| C | `dv`, `th`, `gg`, `am` |

| Tool | When |
|------|------|
| `emit_scripts(...)` | Always, for an accepted Latin display name. Required fields are exactly the tags for the current pass (or the single missing tag on mini-retry). |

**Scripts and codes:**

| Code | Script / language space |
|------|-------------------------|
| `cy` | Cyrillic (Russian, Ukrainian, Bulgarian, Serbian, …) |
| `gk` | Greek |
| `ab` | Arabic script (Arabic, Persian, Urdu, …) |
| `cn` | Chinese Simplified (phonetic / pinyin-aware) |
| `jp` | Japanese (kanji for Japanese names; katakana for foreign) |
| `kr` | Korean Hangul (phonetic) |
| `dv` | Devanagari (Hindi, Sanskrit, Marathi, …) |
| `hb` | Hebrew |
| `th` | Thai |
| `gg` | Georgian |
| `am` | Armenian |

**Intentions for the agent:**

1. Call `emit_scripts` once per pass with exactly one form per required script field.
2. Prefer the most common phonetic convention for that script. Use a well-known conventional form only when one clearly exists (e.g. Einstein → `爱因斯坦`).
3. Follow the script’s native word-order convention where it matters—especially CJK family-name-first when both parts are known.
4. For `cn`, emit Simplified Chinese only; Traditional forms are added in post-processing.
5. For `jp`: primary form in `jp` (kanji, or katakana alone for foreign-only names). When Japanese or ambiguous, also set `jp_kana` to the katakana (or hiragana) form; leave `jp_kana` empty for foreign-only names.
6. Every value must use characters of the target script only.

Shape examples in the system prompt (Guillaume Filion) illustrate argument values. Post-processing treats few-shot CJK strings as contaminants when they appear under a different Latin label, so the agent must transliterate the *current* name, not copy examples.

If any of the eleven scripts is still missing after the three passes (no tool call, empty, or filtered for wrong-script / Latin leak), one mini-retry per missing tag asks for that tag alone—with the rejection hint and shape example—using higher sampling temperature. Clusters still incomplete after those mini-retries are dropped.

---

## Variants available per script

The generator does not aim for a single universal romanization standard. It aims for forms that look like what a literate reader (or a patent clerk) would actually write. Within-script multiplicity comes from deterministic Latin expansion and script-specific post-process expanders (`cn`/`kr`/`cy`/`gk`/`hb`); the transliteration LLM supplies one seed form per script (plus optional `jp_kana`).

### Latin (`la`) — richest deterministic set

Western bibliographic practice only:

| Axis | Variants |
|------|----------|
| Word order | `Given Family`; `Family, Given` (no Eastern family-first without a comma) |
| Given reduction | full; first token; all initials; first + rest-as-initials; hyphenated → `J.-M.` / `J-M` / `JM` |
| Case | Title Case; ALL CAPS; particles `van`/`de`/`von`/… also lowercased |
| Diacritics | keep; strip (`ł`→`l`, `ø`→`o`, `æ`→`ae`, …); German/Nordic digraph (`ü`→`ue`, `ø`→`oe`, `ß`→`ss`) |
| Compounds | hyphen or space (not glued `GarciaLopez`) |
| Korean surnames | `Lee`/`Li`/`Yi`/…, `Park`/`Pak`/`Bak`, `Choi`/`Choe`/`Chey`, `Kim` |
| Arabic prefixes | `Al-`/`El-` attach, space, glue, or drop when already present |

Korean romanization chaos and Arabic article spelling live on this axis because that is where USPTO/EPO fields vary. They are not duplicated under `kr{}` or `ab{}`.

### Chinese (`cn`) — one LLM seed + OpenCC + order flips

- **From the LLM:** one Simplified phonetic (or conventional) Han form; typically middot (`·`) between foreign name parts; native family-first order when both parts are known.
- **Post-process (deterministic):** OpenCC Simplified↔Traditional on every form (e.g. `刘泽东` ↔ `劉澤東`, `张` ↔ `張`)—the main within-script axis for Chinese, analogous to Latin diacritics. For short compact names (2–3 Han characters, no separators), also emit a one-character family-order flip; length-4 compact strings are left alone so two-character surnames are not mangled; two-token forms split by space/middot are swapped.

Traditional is never requested from the model.

### Japanese (`jp`) — LLM only (no order post-process)

- Japanese or ambiguous names: `jp` holds kanji and `jp_kana` holds katakana (or hiragana)—so JPO-style kanji and kana surfaces can co-occur in the cluster.
- Foreign-only names: katakana alone in `jp` (often with `・` separators); leave `jp_kana` empty.
- Family-first order is left to the model’s native-convention rule. Unlike `cn`/`kr`, there is **no** deterministic word-order flip for short `jp` strings.

### Korean (`kr`) — one Hangul seed + order flips; romanization elsewhere

- **From the LLM:** one Hangul phonetic form; family-first when both parts are known. Hangul spelling of a given name is treated as essentially one form (`이지수` is `이지수`).
- **Post-process:** same short-name / two-token order flips as Chinese.
- **Not under `kr{}`:** Lee/Li/Yi-style Latin aliases—those are generated only as `la{...}`.

### Arabic (`ab`) — one LLM seed; Latin carries prefix noise

- **From the LLM:** one Arabic-script rendering; may drop or reshape the article to match native spelling (`فاطمة الحسن` rather than a letter-for-letter Latin mirror).
- **Not expanded in `ab{}`:** Al-/El- attach/space/drop variants—those are Latin post-extraction expansions. Optional patronymic or tribal components that may or may not appear in records are also not systematically varied.

### Cyrillic (`cy`) — one LLM seed + FIO abbreviation/order expansion

- **From the LLM:** one phonetic Cyrillic form; typically Western given–family order (`Иван Иванов`). Occasional three-part FIO when a patronymic is present.
- **Post-process (deterministic):** parse 2-/3-token forms into имя / отчество / фамилия (patronymic detected by suffixes `-ович`/`-евич`/`-овна`/`-евна`/…), then emit the common Russian bibliographic surfaces:
  1. `Фамилия Имя Отчество` (or `Фамилия Имя`)
  2. `Фамилия И.О.` (or `Фамилия И.`)
  3. `Имя Отчество Фамилия` (or `Имя Фамилия`)
  4. `И.О. Фамилия` (or `И. Фамилия`)
  5. `Фамилия, И.О.` (or `Фамилия, И.`)
  Unparseable strings are left unchanged. Title Case only—no ALL CAPS / ё↔е axis.

### Greek (`gk`) — one LLM seed + given/family abbreviation/order expansion

- **From the LLM:** one phonetic Greek form; typically Western given–family order (`Γιάννης Παπαδόπουλος`).
- **Post-process (deterministic):** parse exactly two tokens into given / family, then emit:
  1. `Family Given`
  2. `Family G.`
  3. `Given Family`
  4. `G. Family`
  5. `Family, G.`
  Unparseable strings are left unchanged. Title Case only.

### Hebrew (`hb`) — one LLM seed + given/family abbreviation/order expansion

- **From the LLM:** one phonetic Hebrew form; typically Western given–family order (`יוסף כהן`).
- **Post-process (deterministic):** parse exactly two tokens into given / family, then emit (initials use geresh `׳`):
  1. `Family Given`
  2. `Family G׳`
  3. `Given Family`
  4. `G׳ Family`
  5. `Family, G׳`
  Unparseable strings are left unchanged. No case axis (Hebrew is caseless).

### Other scripts (`dv`, `th`, `gg`, `am`)

One phonetic seed each into the target orthography, typically Western given–family order unless local convention clearly differs. No deterministic within-script expanders. Unicode block checks log mismatches (e.g. Latin leaking into `dv`) and drop bad forms; they do not invent replacements.

---

## Phase 2: alt-Latin augmentation (`generate_names_phase_2.py`)

Phase 2 is a second pass **over phase 1's output**, run separately and later — it does not re-read the raw candidate-string input.

**Input.** Phase 1's cluster output: one tab-separated `tag{value}` line per inventor, the same format described above.

**Output.** The same lines, `la{}` cells augmented for names judged non-Western and left byte-identical otherwise. All eleven non-Latin cells are always passed through unchanged, in their original order. A line that fails to parse as cluster-corpus format (or has no `la{}` cells, or its family name can't be recovered) is written through unmodified rather than dropped — phase 2 never destroys phase 1's work.

**Pipeline:**

1. **Parse.** Split each line into ordered `(tag, value)` cells. Recover `(given, family)` from the `la{}` cells already present — specifically from a comma-form entry (`Family, Given`), since the family name may be multi-token and a naive last-token split of the `Given Family` form would misplace the boundary.
2. **Judge + alt-Latin.** Tool call `emit_alt_latin(alts)`: given the display name `Given Family`, decide from the **family name's** shape whether the person is likely non-Western (Slavic `-ov/-ev/-sky/-enko`, CJK, Arabic `al-/bin/ibn`, Indic, …) — deliberately not from the given name alone, since a given name like "Viktor" also occurs on ordinary Western people (a Hungarian, a German). Only for a family name that reads non-Western does the model also fork well-known cross-language given-name spellings (Viktor→Victor, Yuriy/Jurij→Yuri/Youri, Zhang→Chang, …). Ordinary Western names (`Victor Dupont`, `Ernst A. Mayr`) get `alts=[]`. One retry with pointed feedback if no tool call appears.
3. **Merge.** `latin_variants_with_alts(given, family, alts, base=la_forms)` expands each alt through the same deterministic Latin axes as phase 1 (order, initials, case, diacritics) and appends only the forms not already present, onto the phase-1 `la{}` list rather than recomputing it from scratch. `alts=[]` is therefore a no-op: the `la{}` list — and the whole line — comes out identical to phase 1's.

Example: a phase-1 cluster for `Viktor T. Skokov` (Slavic family name `Skokov`) gains `la{Victor T. Skokov}`, `la{Skokov, Victor T.}`, and their case/initial variants; its `cy{}`/`gk{}`/… cells are untouched. A cluster for `Victor Dupont` (Western family name `Dupont`) is written back unchanged — `Viktor Dupont` is never offered.

---

## What this data is for

Each output line is a positive cluster: many surface forms of one person identity. Downstream Odin training can treat members of a cluster as equivalent (or near-equivalent) spellings across scripts and orthographic conventions. Coverage is uneven by design: Latin and Chinese get the densest within-script axes (structure/diacritics/romanization aliases; Simplified↔Traditional + order flips); Cyrillic/Greek/Hebrew get short FIO or given–family abbreviation/order expanders; Japanese leans on dual kanji/kana via `jp` + optional `jp_kana`; Korean and Arabic database noise is mostly absorbed into `la{}` rather than into Hangul or Arabic script.

Quality controls that protect that intent:

- reject non-person lines early
- require all eleven scripts before writing
- strip few-shot contaminants
- warn on wrong-script characters
- keep Latin (and cn/kr/cy/gk/hb post-process) expansion deterministic so the LLM emits one seed per script only
