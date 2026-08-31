"""Tests for the frozen C tokenizer (``odin_tokenizer_fast``).

Requires the built extension (``make tokenizer-c``) and the real tokenizer
directory; both are checked via skip.
"""

from __future__ import annotations

import json
import random
import string
import sys
from pathlib import Path

import numpy as np
import pytest

REAL_TOKENIZER_DIR = Path("/mnt/nvme1/odin_tokenizer")
TOKENIZER_JSON = REAL_TOKENIZER_DIR / "tokenizer.json"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from odin_tokenizer_fast import OdinFastTokenizer  # noqa: E402

ADVERSARIAL = [
    "J. Smith",
    "Juan Carlos Martinez Anton",
    "O'Brien",
    "O'Neil's",
    "JUAN CARLOS MARTINEZ ANTON",
    "Martinez Anton, Juan Carlos",
    "123 Main St.",
    "No. 5",
    "a  b",
    "a   b",
    "a     b",
    "Bong   Kyo Jong",
    "  leading",
    "trailing  ",
    "a b  c d",
    "  x",
    "x  ",
    " ",
    "  ",
    "a\tb",
    "e\u0301",
    "ı",
    "ß",
    "µ",
    "Œ",
    "ñ",
    "’",
    "J.smith",
    "J. Smith",
    "İ",
    "Ж-и",
    "ж-и",
    "김철수",
    "王 芳",
    "a\u3000b",
    "\U0001f600",
    "[PAD]",
    "[UNK]",
    "[BOS]",
    "[EOS]",
    "[SEP]",
    "[la]",
    "[am]",
    "[la]J. Smith",
    "x[la]y",
    "[la][gk]",
    "a[",
    "a][",
    "a[am",
    "[am",
    "[a",
    "a[9]",
    "\x00\x01\x7f",
    "\x80\xad",
    "a\u0085b",
    "a\u00a0b",
    "a\u2000b",
    "a\u1680b",
    "a\u2028b",
]

_ALPHABET = string.ascii_letters + " .,'’-1234567890" + "éèêëàâäùûüîïôöçßãõáíóúü" + "аеоцжюяі" + "한가"


def _random_corpus(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        length = rng.randint(1, 24)
        s = "".join(rng.choice(_ALPHABET) for _ in range(length))
        if rng.random() < 0.1:
            s = s.replace("a", "[la]", 1) if "a" in s else "[gk]" + s
        out.append(s)
    return out


@pytest.fixture(scope="module")
def fast() -> OdinFastTokenizer:
    if not TOKENIZER_JSON.is_file():
        pytest.skip("real tokenizer not mounted")
    return OdinFastTokenizer(TOKENIZER_JSON)


@pytest.fixture(scope="module")
def hf():
    if not TOKENIZER_JSON.is_file():
        pytest.skip("real tokenizer not mounted")
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast.from_pretrained(str(REAL_TOKENIZER_DIR))  # nosec: B615  # local dir, not the Hub


def test_attributes(fast):
    assert fast.vocab_size == 16384
    assert fast.pad_token_id == 0
    assert fast.unk_token_id == 1
    assert fast.bos_token_id == 2
    assert fast.eos_token_id == 3


def test_encode_matches_hf(fast, hf):
    corpus = ADVERSARIAL + _random_corpus(300, seed=7)
    for s in corpus:
        assert fast.encode(s) == hf.encode(s, add_special_tokens=False), s


def test_encode_many_matches_hf(fast, hf):
    corpus = ADVERSARIAL + _random_corpus(100, seed=11)
    mine = fast.encode_many(corpus)
    want = [hf.encode(s, add_special_tokens=False) for s in corpus]
    assert mine == want
    assert fast.encode_many([]) == []


def test_encode_padded(fast, hf):
    corpus = ["J. Smith", "Juan Carlos Martinez Anton", "x", "a  b  c", "김철수"]
    ids, lens = fast.encode_padded(corpus, 12)
    assert ids.shape == (len(corpus), 12)
    assert ids.dtype == np.uint16
    assert lens.dtype == np.uint16
    for i, s in enumerate(corpus):
        want = hf.encode(s, add_special_tokens=False)
        clipped = min(len(want), 12)
        assert lens[i] == len(want)  # raw (untruncated) count
        assert ids[i, :clipped].tolist() == want[:clipped]
        assert (ids[i, clipped:] == fast.pad_token_id).all()


def test_decode_matches_hf(fast, hf):
    corpus = ADVERSARIAL + _random_corpus(100, seed=13)
    for s in corpus:
        ids = hf.encode(s, add_special_tokens=False)
        assert fast.decode(ids, skip_special_tokens=True) == hf.decode(ids, skip_special_tokens=True), s
        assert fast.decode(ids, skip_special_tokens=False) == hf.decode(ids, skip_special_tokens=False), s


def test_decode_specials_semantics(fast):
    # specials are dropped by default, rendered literally with skip=False
    assert fast.decode([5, 0]) == ""
    assert fast.decode([5, 58, 30], skip_special_tokens=False) == "[la]" + fast.decode(
        [58, 30], skip_special_tokens=False
    )
    assert fast.decode_many([[5, 0], [3, 5]], skip_special_tokens=False) == ["[la][PAD]", "[EOS][la]"]


def test_lossless_roundtrip(fast):
    for s in ADVERSARIAL + _random_corpus(200, seed=17):
        ids = fast.encode(s)
        if any(i < 17 for i in ids):
            continue  # special tokens are dropped by the default decode
        assert fast.decode(ids) == s, s


def test_special_extraction(fast):
    assert fast.encode("[la]") == [5]
    assert fast.encode("x[la]y") == fast.encode("x") + [5] + fast.encode("y")
    assert fast.encode("[la][gk]") == [5, 7]
    # near-misses are NOT special tokens
    for s in ["[a", "[am", "a[am", "[9]", "a["]:
        assert all(i >= 17 for i in fast.encode(s))


def test_convert_tokens_to_ids(fast, hf):
    tags = ["la", "cy", "gk", "ab", "cn", "jp", "kr", "dv", "hb", "th", "gg", "am"]
    for tok in ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]"] + [f"[{t}]" for t in tags]:
        assert fast.convert_tokens_to_ids(tok) == hf.convert_tokens_to_ids(tok)
    pj = json.loads(TOKENIZER_JSON.read_text())
    vocab = pj["model"]["vocab"]
    for tok, tid in list(vocab.items())[:200]:
        assert fast.convert_tokens_to_ids(tok) == tid
    assert fast.convert_tokens_to_ids("definitely_not_a_token_xyz") == fast.unk_token_id
    assert fast._c.token_to_id("definitely_not_a_token_xyz") is None


def test_convert_ids_to_tokens(fast, hf):
    for tok, tid in list(json.loads(TOKENIZER_JSON.read_text())["model"]["vocab"].items())[:100]:
        assert fast.convert_ids_to_tokens([tid]) == tok


def test_cache_consistency(fast):
    # repeated + interleaved encodes must always agree with HF (word-cache races)
    from transformers import PreTrainedTokenizerFast

    hf = PreTrainedTokenizerFast.from_pretrained(str(REAL_TOKENIZER_DIR))  # nosec: B615  # local dir, not the Hub
    a, b = "Juan Carlos Martinez", "Juana Carolina Martin"
    for _ in range(50):
        assert fast.encode(a) == hf.encode(a, add_special_tokens=False)
        assert fast.encode(b) == hf.encode(b, add_special_tokens=False)
    assert fast.encode(a) == fast.encode_many([a] * 64)[0]


def test_threading(fast):
    corpus = _random_corpus(2000, seed=23)
    fast.set_num_threads(1)
    single = fast.encode_many(corpus)
    fast.set_num_threads(4)
    multi = fast.encode_many(corpus)
    assert single == multi
    fast.set_num_threads(8)
    assert fast.encode_many(corpus) == single
    fast.set_num_threads(1)
