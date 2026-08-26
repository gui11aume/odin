"""Tests for the Odin BPE tokenizer trainer (src/train_odin_tokenizer.py)."""

from __future__ import annotations

import importlib.util
import unicodedata
from pathlib import Path

from transformers import PreTrainedTokenizerFast

_SCRIPT = Path(__file__).resolve().parents[1] / "src" / "train_odin_tokenizer.py"
_spec = importlib.util.spec_from_file_location("train_odin_tokenizer", _SCRIPT)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

SPECIAL_TOKENS = _mod.SPECIAL_TOKENS
TAGS = _mod.TAGS
build_tokenizer = _mod.build_tokenizer
save_tokenizer = _mod.save_tokenizer
train = _mod.train
verify = _mod.verify
vocab_report = _mod.vocab_report

LINE_1 = "la{John Smith}\tla{JOHN SMITH}\tla{Smith, John}\tcy{Джон Смит}\tcn{约翰·史密斯}"
LINE_2 = "la{José María García-López}\tla{J. GARCIA}\tgg{ხოსე მარია}\tjp{ホセ・マリア・ガルシア}"
LINE_3 = "la{Ernst A. Mayr}\tla{Mayr, Ernst A.}\tam{Էռնստ Մայր}"
CORPUS = "\n".join([LINE_1, LINE_2, LINE_3] * 30) + "\n"
UNSEEN = "Zqzx Ümlaut 未知字符 ꫴ"


def _trained(tmp_path: Path, *, vocab_size: int = 512, corpus: str = CORPUS):
    src = tmp_path / "corpus.txt"
    src.write_text(corpus, encoding="utf-8")
    tok = build_tokenizer()
    train(tok, src, vocab_size=vocab_size)
    return src, tok


def test_train_and_roundtrip_lossless_including_unseen(tmp_path: Path) -> None:
    _, tok = _trained(tmp_path)
    for s in [
        "John Smith",
        "JOHN SMITH",
        "约翰·史密斯",
        "ホセ・マリア・ガルシア",
        "García-López, José María",
        UNSEEN,
    ]:
        enc = tok.encode(s)
        assert tok.decode(enc.ids) == s, f"roundtrip failed for {s!r}"


def test_roundtrip_is_byte_exact_on_non_nfc_input(tmp_path: Path) -> None:
    # The corpus contains canonically decomposed cells (decomposed Greek
    # Ϊ/tonos, Turkish I+dot, Devanagari candrabindu) and real-world input
    # does too. The tokenizer must not rewrite them (no NFC normalizer).
    _, tok = _trained(tmp_path)
    non_nfc = [
        "Τσαν Χο-" + chr(0x0399) + chr(0x0308) + chr(0x0301) + "ν",  # decomposed: IOTA + dialytika + tonos
        "ANIL VEL" + chr(0x0049) + chr(0x0307) + " ALPAN",  # decomposed: LATIN I + dot above
        chr(0x092E)
        + chr(0x0930)
        + chr(0x0935)
        + chr(0x093E)
        + chr(0x0928)
        + chr(0x093C)
        + chr(0x0020)
        + chr(0x0905)
        + chr(0x0932)
        + chr(0x092C)
        + chr(0x0915)
        + chr(0x093E)
        + chr(0x0932)
        + chr(0x0940),  # decomposed: NA + candrabindu
    ]
    for s in non_nfc:
        assert s != unicodedata.normalize("NFC", s), f"test fixture {s!r} is NFC"
        assert tok.decode(tok.encode(s).ids) == s


def test_vocab_has_no_container_syntax(tmp_path: Path) -> None:
    _, tok = _trained(tmp_path)
    report = vocab_report(tok)
    assert report["pieces_with_braces"] == []
    assert report["missing_specials"] == []


def test_special_tokens_and_vocab_size(tmp_path: Path) -> None:
    _, tok = _trained(tmp_path, vocab_size=256)
    vocab = tok.get_vocab()
    assert all(t in vocab for t in SPECIAL_TOKENS)
    # Floor is the 256 byte-alphabet pieces + the special tokens; no merges fit.
    assert len(vocab) == 256 + len(SPECIAL_TOKENS)
    assert tok.token_to_id("[UNK]") is not None


def test_trains_on_values_not_raw_cells(tmp_path: Path) -> None:
    # The raw container syntax must not be learned: encoding a cell with its
    # tag{...} wrapper differs from encoding the bare value, and no brace
    # piece may exist (checked here and via vocab_report).
    _, tok = _trained(tmp_path)
    assert tok.encode("la{John Smith}").ids != tok.encode("John Smith").ids
    assert vocab_report(tok)["pieces_with_braces"] == []


def test_train_counts_cells_and_max_clusters_limits(tmp_path: Path) -> None:
    src = tmp_path / "corpus.txt"
    src.write_text(CORPUS, encoding="utf-8")
    tok = build_tokenizer()
    assert train(tok, src, vocab_size=256) == 30 * 12  # 12 cells per cluster cycle
    tok2 = build_tokenizer()
    assert train(tok2, src, vocab_size=256, max_clusters=3) == 12  # one full cycle


def test_verify_reports_per_tag_and_zero_failures(tmp_path: Path) -> None:
    src, tok = _trained(tmp_path)
    stats = verify(tok, src, chunk_size=100)
    assert stats["total_cells"] == 360
    assert stats["roundtrip_failures"] == 0
    assert stats["unk_tokens"] == 0
    assert stats["per_tag"]["la"][1] == 30 * 7
    assert stats["per_tag"]["cn"][1] == 30
    assert all(stats["per_tag"][t][0] > 0 for t in ("la", "cy", "cn", "gg", "jp", "am"))


def test_save_and_reload_hf_roundtrip(tmp_path: Path) -> None:
    _, tok = _trained(tmp_path)
    out = tmp_path / "tok"
    save_tokenizer(tok, out)
    assert (out / "tokenizer.json").is_file()
    assert (out / "tokenizer_config.json").is_file()

    loaded = PreTrainedTokenizerFast.from_pretrained(str(out))  # nosec: B615  # local dir, not the Hub
    assert loaded.pad_token == "[PAD]"
    assert loaded.eos_token == "[EOS]"
    assert loaded.decode(loaded.encode("约翰·史密斯", add_special_tokens=False)) == "约翰·史密斯"
    assert loaded.decode(loaded.encode(UNSEEN, add_special_tokens=False)) == UNSEEN
