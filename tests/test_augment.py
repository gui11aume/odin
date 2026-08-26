"""Tests for letter-level corruption (augment)."""

from __future__ import annotations

import random

import pytest

from odin_vae.augment import (
    CASE_SCRIPTS,
    CJK_SCRIPTS,
    CORRUPTIBLE_SCRIPTS,
    LATIN_SCRIPT,
    SCRIPTS,
    LetterAugmenter,
    _merge_groups,
)

LATIN_TABLE = {LATIN_SCRIPT: [("a", 5.0), ("b", 3.0), ("c", 1.0)]}
CY_TABLE = {"cy": [("\u0430", 5.0), ("\u0431", 1.0)]}  # а б


def test_script_constants() -> None:
    assert len(SCRIPTS) == 12
    assert CJK_SCRIPTS == {"cn", "jp", "kr"}
    assert CORRUPTIBLE_SCRIPTS == set(SCRIPTS) - CJK_SCRIPTS
    assert CASE_SCRIPTS <= CORRUPTIBLE_SCRIPTS


def test_cjk_never_corrupted() -> None:
    aug = LetterAugmenter(rate=1.0)
    for script in CJK_SCRIPTS:
        text = {"cn": "\u5f20\u4e09", "jp": "\u5c71\u5c71", "kr": "\uae40\uae00"}[script]
        assert aug.corrupt(script, text, random.Random(0)) == text


def test_rate_zero_is_identity() -> None:
    aug = LetterAugmenter(rate=0.0)
    assert aug.corrupt("la", "John Smith", random.Random(0)) == "John Smith"
    assert aug.corrupt("ab", "\u0645\u062d\u0645\u062f", random.Random(0)) == "\u0645\u062d\u0645\u062f"


def test_rate_one_corrupts_every_letter() -> None:
    aug = LetterAugmenter(rate=1.0, letter_frequencies=LATIN_TABLE)
    out = aug.corrupt("la", "defgh", random.Random(42))
    # Every letter either deleted or substituted: no original letter survives
    # (d/e/f/h have no confusion class and the table only holds a/b/c;
    # g only confuses with 9/G).
    for ch in "defgh":
        assert ch not in out


def test_single_letter_never_deleted() -> None:
    aug = LetterAugmenter(rate=1.0, letter_frequencies=LATIN_TABLE)
    for seed in range(50):
        out = aug.corrupt("la", "A", random.Random(seed))
        assert len(out) == 1  # substituted, never deleted


def test_corruption_stays_in_alphabet_plus_digits() -> None:
    aug = LetterAugmenter(rate=1.0, letter_frequencies=LATIN_TABLE)
    out = aug.corrupt("la", "abcdefgh", random.Random(7))
    allowed = set("abcABC0123456789") | set("abcdefgh")
    assert set(out) <= allowed


def test_case_pairs_added_automatically() -> None:
    aug = LetterAugmenter(rate=0.0, letter_frequencies={LATIN_SCRIPT: [("a", 1.0), ("A", 1.0)]})
    assert "A" in aug.classes[LATIN_SCRIPT]["a"]
    assert "a" in aug.classes[LATIN_SCRIPT]["A"]
    # Scripts without case in the table get no spurious case pairs.
    assert "la" not in CJK_SCRIPTS


def test_confusion_pairs_symmetric() -> None:
    aug = LetterAugmenter(rate=0.0, letter_frequencies=CY_TABLE)
    cy = aug.classes["cy"]
    # Cyrillic а (U+0430) and latin a are mutually confusable.
    assert "a" in cy["\u0430"]
    assert "\u0430" in cy["a"]
    # Transitivity: o ~ 0 (group "o0") and O ~ 0 (group "O0") => o ~ O.
    aug2 = LetterAugmenter(rate=0.0, letter_frequencies={LATIN_SCRIPT: [("o", 1.0), ("O", 1.0)]})
    assert "O" in aug2.classes[LATIN_SCRIPT]["o"]
    assert "o" in aug2.classes[LATIN_SCRIPT]["O"]


def test_confusion_branch_respected() -> None:
    # With confusion_weight=0 the confusion class is never used: substitutions
    # come only from the frequency table.
    aug = LetterAugmenter(rate=1.0, letter_frequencies=LATIN_TABLE, confusion_weight=0.0)
    seen: set[str] = set()
    for seed in range(200):
        out = aug.corrupt("la", "e", random.Random(seed))  # e has no confusion class
        seen.add(out)
    assert seen <= {"a", "b", "c"}


def test_confusion_branch_used() -> None:
    aug = LetterAugmenter(rate=1.0, letter_frequencies=LATIN_TABLE, confusion_weight=1.0)
    # 'o' confuses with 0/O; 'l' with 1/I; etc.
    outs = {aug.corrupt("la", "o", random.Random(s)) for s in range(60)}
    assert outs <= {"0", "O"}
    assert "0" in outs


def test_deletion_and_substitution_both_happen() -> None:
    aug = LetterAugmenter(rate=1.0, letter_frequencies=LATIN_TABLE)
    lengths: set[int] = set()
    for seed in range(200):
        out = aug.corrupt("la", "abcd", random.Random(seed))
        lengths.add(len(out))
    assert min(lengths) < 4  # some deletions
    assert max(lengths) == 4  # no insertions


def test_non_alpha_chars_untouched() -> None:
    aug = LetterAugmenter(rate=1.0, letter_frequencies=LATIN_TABLE)
    out = aug.corrupt("la", "J. D. 1985", random.Random(3))
    # Spaces, dots and digits are never modified.
    assert out.count(" ") == 2
    assert out.count(".") == 2
    assert "1985" in out


def test_merge_groups_transitive() -> None:
    merged = _merge_groups(("ab", "bc", "de"))
    assert merged["a"] == ("b", "c")
    assert merged["b"] == ("a", "c")
    assert merged["c"] == ("a", "b")
    assert merged["d"] == ("e",)


def test_invalid_parameters_rejected() -> None:
    with pytest.raises(ValueError, match="rate"):
        LetterAugmenter(rate=1.5)
    with pytest.raises(ValueError, match="confusion_weight"):
        LetterAugmenter(rate=0.1, confusion_weight=1.5)
