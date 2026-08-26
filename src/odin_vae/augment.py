"""Letter-level corruption for Odin name surfaces.

The corruption mirrors small transcription/OCR errors in patent data: with
probability ``rate`` per letter, the letter is either deleted or substituted.
Substitutions are drawn from a script-specific visual-confusion class with
probability ``confusion_weight`` (e.g. Cyrillic ``а``/latin ``a``, Arabic
``ب``/``ت``) and from the corpus letter-frequency table otherwise.

CJK scripts (cn/jp/kr) are never corrupted: every character there is a full
semantic unit, and the natural variant diversity for those scripts is already
present in the cluster data.
"""

from __future__ import annotations

import random
from bisect import bisect_left
from collections.abc import Iterable, Mapping, Sequence

# The twelve script tags of the cluster corpus.
SCRIPTS: tuple[str, ...] = ("la", "cy", "gk", "ab", "cn", "jp", "kr", "dv", "hb", "th", "gg", "am")
LATIN_SCRIPT: str = "la"
CJK_SCRIPTS: frozenset[str] = frozenset({"cn", "jp", "kr"})
CORRUPTIBLE_SCRIPTS: frozenset[str] = frozenset(SCRIPTS) - CJK_SCRIPTS
# Scripts with a distinct uppercase alphabet: case pairs are added to the
# confusion classes automatically from the letter-frequency table.
CASE_SCRIPTS: frozenset[str] = frozenset({"la", "cy", "gk", "am"})

# Visual-confusion groups per script. Each group is a string of mutually
# confusable code points. Non-ASCII code points are written as escapes with
# the glyph in a trailing comment.
CONFUSION_GROUPS: dict[str, tuple[str, ...]] = {
    # Latin: letter/digit OCR pairs (plus l/I/i, g/q).
    "la": ("o0", "O0", "l1Il", "i1", "S5", "s5", "B8", "b6", "g9G", "q9", "z2Z", "T7t"),
    # Cyrillic: cross-script lookalikes (latin + greek) and intra-script pairs.
    "cy": (
        "\u0430a",  # а а
        "\u0435e",  # е e
        "\u043eo\u03bf",  # о о ο
        "\u0441c",  # с c
        "\u0440p\u03c1",  # р p ρ
        "\u0432B",  # в B
        "\u043dHn",  # н H n
        "\u043aK",  # к K
        "\u043cM",  # м M
        "\u0442T7",  # т T 7
        "\u0443y",  # у y
        "\u0445X",  # х X
        "\u0434D6",  # д D 6
        "\u0456i",  # і i
        "\u0458j",  # ј j
        "\u04373",  # з 3
        "\u0431\u0432B\u0412",  # б в B В
    ),
    # Greek (lowercase names): cross-script lookalikes.
    "gk": (
        "\u03b1a",  # α a
        "\u03bfo",  # ο o
        "\u03b2B",  # β B
        "\u03b3y",  # γ y
        "\u03b9i",  # ι i
        "\u03bak",  # κ k
        "\u03bcm",  # μ m
        "\u03bdv",  # ν v
        "\u03c1p",  # ρ p
        "\u03c3s",  # σ s
        "\u03c2c",  # ς c
        "\u03c4t",  # τ t
        "\u03c6f",  # φ f
        "\u03c7x",  # χ x
        "\u03c0n",  # π n
    ),
    # Arabic: dot-pattern and shape confusions (no case).
    "ab": (
        "\u0628\u062a",  # ب ت
        "\u062a\u062b",  # ت ث
        "\u062c\u062d",  # ج ح
        "\u062e\u062f\u0630",  # خ د ذ
        "\u0631\u0632",  # ر ز
        "\u0633\u0634",  # س ش
        "\u0645\u0646",  # م ن
        "\u0647\u0629",  # ه ة
        "\u0627\u0623\u0625\u0622",  # ا أ إ آ
        "\u064a\u0649",  # ي ى
    ),
    # Hebrew: final forms and shape confusions (no case).
    "hb": (
        "\u05db\u05da",  # כ ך
        "\u05de\u05dd",  # מ ם
        "\u05e3\u05e4",  # פ ף
        "\u05e5\u05e6",  # צ ץ
        "\u05d3\u05e8",  # ד ר
    ),
    # Devanagari: dot/stroke-variation chains of the consonant blocks.
    "dv": (
        "\u0915\u0916",  # क ख
        "\u0916\u0917",  # ख ग
        "\u0917\u0918",  # ग घ
        "\u091a\u091c",  # च छ
        "\u091c\u091d",  # छ ज
        "\u091d\u091e",  # ज झ
        "\u091f\u0920",  # ट ठ
        "\u0920\u0921",  # ठ ड
        "\u0921\u0922",  # ड ढ
        "\u0924\u0925",  # त थ
        "\u0925\u0926",  # थ द
        "\u0926\u0927",  # द ध
        "\u092a\u092b",  # प फ
        "\u092c\u092d",  # ब भ
        "\u0936\u0937",  # श ष
    ),
    # Thai: shape confusions (no case).
    "th": (
        "\u0e14\u0e15",  # ด ต
        "\u0e1e\u0e1f",  # พ ฟ
        "\u0e1f\u0e20",  # ฟ ภ
        "\u0e2d\u0e19",  # ฐ ฏ
        "\u0e02\u0e0a",  # ข ช
        "\u0e2a\u0e28",  # ส ศ
        "\u0e22\u0e23",  # ย ร
    ),
    # Georgian (Mkhedrulu): thin confusion set.
    "gg": (
        "\u10d6\u10df",  # ზ ჟ
        "\u10df\u10e0",  # ჟ რ
        "\u10e0\u10e6",  # რ ღ
        "\u10de\u10e4",  # პ ფ
    ),
    # Armenian: thin confusion set (case pairs added automatically).
    "am": (
        "\u0532\u0533",  # Բ Գ
        "\u0539\u0549",  # Թ Չ
        "\u0535\u0537",  # Ե Է
    ),
}


def _merge_groups(groups: Iterable[str]) -> dict[str, tuple[str, ...]]:
    """Merge overlapping confusion groups into a letter -> class-mates map.

    Groups are merged transitively (``ab`` + ``bc`` puts ``a`` and ``c`` in
    the same class). The result maps every letter to its sorted class-mates.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for group in groups:
        for char in group[1:]:
            if group[0] not in parent:
                parent[group[0]] = group[0]
            if char not in parent:
                parent[char] = char
            ra, rb = find(group[0]), find(char)
            if ra != rb:
                parent[rb] = ra

    classes: dict[str, set[str]] = {}
    for x in parent:
        classes.setdefault(find(x), set()).add(x)
    return {x: tuple(sorted(cls - {x})) for cls in classes.values() for x in cls}


class LetterAugmenter:
    """Letter-level corruption (deletion / substitution) per script.

    Args:
        rate: Probability of corrupting each letter (0 disables).
        letter_frequencies: Optional per-script table of ``(letter, weight)``
            pairs used for the uniform-substitution branch.
        confusion_weight: Probability that a substitution is drawn from the
            visual-confusion class of the original letter.
    """

    def __init__(
        self,
        rate: float,
        letter_frequencies: Mapping[str, Sequence[tuple[str, float]]] | None = None,
        confusion_weight: float = 0.7,
    ):
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"rate must be in [0, 1], got {rate}")
        if not 0.0 <= confusion_weight <= 1.0:
            raise ValueError(f"confusion_weight must be in [0, 1], got {confusion_weight}")
        self.rate = float(rate)
        self.confusion_weight = float(confusion_weight)

        freqs = dict(letter_frequencies or {})
        groups: dict[str, list[str]] = {s: list(CONFUSION_GROUPS.get(s, ())) for s in SCRIPTS}
        for script in CASE_SCRIPTS:
            letters = {c for c, _ in freqs.get(script, ())}
            for c in sorted(letters):
                u = c.swapcase()
                if u != c and u in letters:
                    groups[script].append(c + u)
        self.classes = {s: _merge_groups(g) for s, g in groups.items() if g}

        self.tables: dict[str, tuple[list[str], list[float]]] = {}
        for script, pairs in freqs.items():
            letters = sorted((c, max(0.0, float(w))) for c, w in pairs if c.isalpha() and w > 0.0)
            if letters:
                chars = [c for c, _ in letters]
                weights = [w for _, w in letters]
                cumulative: list[float] = []
                total = 0.0
                for w in weights:
                    total += w
                    cumulative.append(total)
                self.tables[script] = (chars, cumulative)

    def _substitute(self, script: str, char: str, rng: random.Random) -> str:
        class_mates = self.classes.get(script, {}).get(char)
        if class_mates and rng.random() < self.confusion_weight:
            return rng.choice(class_mates)
        table = self.tables.get(script)
        if table is not None:
            chars, cumulative = table
            if cumulative[-1] > 0.0:
                x = rng.random() * cumulative[-1]
                return chars[bisect_left(cumulative, x)]
        return char

    def corrupt(self, script: str, text: str, rng: random.Random) -> str:
        """Corrupt one surface; returns the text unchanged for CJK scripts."""
        if script not in CORRUPTIBLE_SCRIPTS or self.rate == 0.0 or not text:
            return text
        alpha_count = sum(ch.isalpha() for ch in text)
        out: list[str] = []
        for ch in text:
            if ch.isalpha() and rng.random() < self.rate:
                if alpha_count > 1 and rng.random() < 0.5:
                    continue  # deletion (never empties the surface)
                out.append(self._substitute(script, ch, rng))
            else:
                out.append(ch)
        return "".join(out)
