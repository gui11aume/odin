"""Generate the frozen lookup-table header for the C tokenizer.

Reads the HuggingFace ``tokenizer.json`` and emits ``odin_tokenizer_tables.h``
containing everything the C encoder/decoder needs:

* ``BYTE_ID`` – 256-entry table mapping raw UTF-8 bytes to the single-byte
  token id (the ByteLevel alphabet; tokenizers 0.22.2 ``bytes_char`` mapping).
* ``SPECIAL_*`` – the 17 special tokens ([PAD] ... [am]) for the literal
  leftmost-longest extraction pass.
* ``MERGE_KEY`` / ``MERGE_VAL`` – open-addressing table implementing the
  BPE merge list: key = (left_id << 14) | right_id -> (rank << 16) | new_id.
* ``CL_L/CL_N/CL_WS`` – BMP bitmaps + non-BMP range tables for the
  GPT-2 ByteLevel pre-tokenizer regex (\\p{L} / \\p{N} / \\s classes).
* ``PIECE_OFF`` / ``PIECE_LEN`` / ``PIECE_BYTES`` – byte content of every
  vocab token (id order), for decoding.
* ``TID_*`` – open-addressing table token string -> id (``token_to_id``).

Usage:  python gen_tables.py [TOKENIZER_JSON] [OUT_HEADER]
"""

from __future__ import annotations

import hashlib
import json
import sys
import unicodedata
from pathlib import Path

VOCAB_SIZE = 16384
MERGE_SLOTS = 65536
TID_SLOTS = 65536
ID_BITS = 14  # 2**14 == 16384

# White_Space per Unicode (what the Rust `regex` crate's \s matches)
_WS_EXTRAS = frozenset({0x85, 0xA0, 0x1680, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000} | set(range(0x2000, 0x200B)))
_WS_BASIC = frozenset({0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20})


def build_byte_map() -> dict[int, str]:
    """tokenizers 0.22.2 byte_level.rs ``bytes_char`` (authoritative)."""
    bs = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    b2c = {b: chr(b) for b in bs}
    n = 0
    for b in range(256):
        if b not in b2c:
            b2c[b] = chr(256 + n)
            n += 1
    assert len(b2c) == 256
    return b2c


def classify_bmp() -> tuple[list[int], list[int], list[int]]:
    l_bits, n_bits, ws_bits = [0] * 65536, [0] * 65536, [0] * 65536
    for cp in range(0x10000):
        ch = chr(cp)
        cat = unicodedata.category(ch)
        if cat in ("Lu", "Ll", "Lt", "Lm", "Lo"):
            l_bits[cp] = 1
        elif cat in ("Nd", "Nl", "No"):
            n_bits[cp] = 1
        if cp in _WS_BASIC or cp in _WS_EXTRAS:
            ws_bits[cp] = 1
    return l_bits, n_bits, ws_bits


def classify_nonbmp() -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    l_r = _ranges((0x10000, 0x110000), ("Lu", "Ll", "Lt", "Lm", "Lo"))
    n_r = _ranges((0x10000, 0x110000), ("Nd", "Nl", "No"))
    # Unicode White_Space has no non-BMP members
    w_r: list[tuple[int, int]] = []
    return l_r, n_r, w_r


def _ranges(span: tuple[int, int], cats: tuple[str, ...]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start: int | None = None
    end = span[0] - 1
    for cp in range(span[0], span[1]):
        hit = unicodedata.category(chr(cp)) in cats
        if hit:
            if start is None:
                start = cp
            end = cp
        elif start is not None:
            out.append((start, end))
            start = None
    if start is not None:
        out.append((start, end))
    return out


def fmt_u16(a: list[int]) -> str:
    lines = []
    for i in range(0, len(a), 16):
        lines.append("    " + ", ".join(str(x) for x in a[i : i + 16]) + ("," if i + 16 < len(a) else ""))
    return "\n".join(lines)


def fmt_u32(a: list[int]) -> str:
    lines = []
    for i in range(0, len(a), 16):
        lines.append("    " + ", ".join(f"0x{x:08X}u" for x in a[i : i + 16]) + ("," if i + 16 < len(a) else ""))
    return "\n".join(lines)


def fmt_u64(a: list[int]) -> str:
    lines = []
    for i in range(0, len(a), 8):
        lines.append("    " + ", ".join(f"0x{x:016X}ULL" for x in a[i : i + 8]) + ("," if i + 8 < len(a) else ""))
    return "\n".join(lines)


def fmt_u8(a: list[int]) -> str:
    lines = []
    for i in range(0, len(a), 32):
        lines.append("    " + ", ".join(str(x) for x in a[i : i + 32]) + ("," if i + 32 < len(a) else ""))
    return "\n".join(lines)


def main() -> None:
    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/mnt/nvme1/odin_tokenizer/tokenizer.json")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent / "odin_tokenizer_tables.h"

    pj = json.loads(json_path.read_text())
    vocab = pj["model"]["vocab"]
    merges = pj["model"]["merges"]
    assert len(vocab) == VOCAB_SIZE, len(vocab)
    max_id = max(vocab.values())
    assert max_id == VOCAB_SIZE - 1

    id2token: list[str] = [None] * VOCAB_SIZE  # type: ignore[misc]
    for k, v in vocab.items():
        assert id2token[v] is None
        id2token[v] = k
    assert None not in id2token

    b2c = build_byte_map()
    c2b = {c: b for b, c in b2c.items()}

    # ---- BYTE_ID ----------------------------------------------------------
    byte_id = [0] * 256
    seen = [False] * 256
    for k, v in vocab.items():
        if len(k) == 1:
            b = c2b[k]
            assert not seen[b], f"duplicate byte {b:#x}"
            byte_id[b] = v
            seen[b] = True
    assert all(seen), "not all 256 bytes have a single-char token"

    # ---- specials ----------------------------------------------------------
    specials = [(t["content"], t["id"]) for t in pj["added_tokens"] if t.get("special")]
    assert len(specials) == 17
    special_ids = sorted(i for _, i in specials)
    assert special_ids == list(range(17)), special_ids
    assert all(4 <= len(c) <= 5 and c[0] == "[" and c[-1] == "]" for c, _ in specials)

    # second-byte -> candidate entries (full tail bytes content[1:] + id)
    sec_index: dict[int, list[tuple[list[int], int]]] = {}
    for content, tid in specials:
        b = [ord(ch) for ch in content]
        sec_index.setdefault(b[1], []).append((b[1:], tid))
    sec_tail: list[tuple[list[int], int]] = []
    for sec in sorted(sec_index):
        sec_tail.extend(sorted(sec_index[sec], key=lambda e: e[0]))
    SPECIAL_SEC = [0xFF] * 256
    SPECIAL_SEC_N = [0] * 256
    off = 0
    for sec in sorted(sec_index):
        SPECIAL_SEC[sec] = off
        SPECIAL_SEC_N[sec] = len(sec_index[sec])
        off += len(sec_index[sec])
    assert off == len(sec_tail) == 17

    # ---- merge map ---------------------------------------------------------
    merge_key = [0] * MERGE_SLOTS
    merge_val = [0] * MERGE_SLOTS
    n_merges = 0
    for rank, (a, b) in enumerate(merges):
        a_id, b_id = vocab[a], vocab[b]
        new_id = vocab[a + b]
        assert rank < 0x10000 and new_id < 0x10000
        key = (a_id << ID_BITS) | b_id
        stored = key + 1
        slot = (key * 0x9E3779B9) & (MERGE_SLOTS - 1)
        while merge_key[slot] != 0:
            assert merge_key[slot] != stored, f"duplicate merge pair {a!r} {b!r}"
            slot = (slot + 1) & (MERGE_SLOTS - 1)
        merge_key[slot] = stored
        merge_val[slot] = (rank << 16) | new_id
        n_merges += 1
    assert n_merges < MERGE_SLOTS // 2

    # ---- unicode classes ----------------------------------------------------
    l_bits, n_bits, ws_bits = classify_bmp()
    l_r, n_r, w_r = classify_nonbmp()
    assert not w_r, f"unexpected non-BMP whitespace: {w_r[:5]}"

    def bits_to_words(bits: list[int]) -> list[int]:
        words = [0] * 8192
        for cp, v in enumerate(bits):
            if v:
                words[cp >> 6] |= 1 << (cp & 63)
        return words

    # ---- pieces (decode) -----------------------------------------------------
    piece_off: list[int] = []
    piece_len: list[int] = []
    piece_bytes: list[int] = []
    for tid in range(VOCAB_SIZE):
        piece_off.append(len(piece_bytes))
        # find the token with this id (id -> token reverse map)
        s = id2token[tid]
        bs_ = bytes(c2b[ch] for ch in s)
        piece_len.append(len(bs_))
        piece_bytes.extend(bs_)
    max_piece = max(piece_len)
    assert max_piece <= 255

    # ---- token_to_id table ----------------------------------------------------
    tid_key = [0] * TID_SLOTS
    tid_hash = [0] * TID_SLOTS
    for tid in range(VOCAB_SIZE):
        raw = bytes(c2b[ch] for ch in id2token[tid])
        h = 2166136261
        for byte in raw:
            h ^= byte
            h = (h * 16777619) & 0xFFFFFFFF
        slot = h & (TID_SLOTS - 1)
        while tid_key[slot] != 0:
            slot = (slot + 1) & (TID_SLOTS - 1)
        tid_key[slot] = tid + 1
        tid_hash[slot] = h

    # ---- checksum (majestic-style, vocab k:v sorted) ---------------------------
    # ---- char->byte table (token_to_id: UTF-8 token chars -> raw piece bytes) --
    CHAR_TO_BYTE = [0xFF] * 352  # covers cp < 0x160 (all byte-mapped chars)
    for b, c in b2c.items():
        CHAR_TO_BYTE[ord(c)] = b

    # ---- checksum (majestic-style, vocab k:v sorted) ---------------------------
    checksum = hashlib.md5(
        "".join(f"{k}{v}" for k, v in sorted(vocab.items())).encode(), usedforsecurity=False
    ).hexdigest()  # nosec: B324  # non-security use: vocab integrity checksum

    # ---- emit ---------------------------------------------------------------
    h = []
    h.append("// GENERATED by gen_tables.py -- DO NOT EDIT.")
    h.append(f"// Source tokenizer: {json_path}")
    h.append(f"// Vocab checksum (md5 of sorted 'k{v}'): {checksum}")
    h.append("#ifndef ODIN_TOKENIZER_TABLES_H")
    h.append("#define ODIN_TOKENIZER_TABLES_H")
    h.append("")
    h.append(f"#define VOCAB_SIZE {VOCAB_SIZE}")
    h.append(f'#define VOCAB_CHECKSUM "{checksum}"')
    h.append("#define N_SPECIALS 17")
    h.append("")
    h.append("static const uint16_t BYTE_ID[256] = {")
    h.append(fmt_u16(byte_id))
    h.append("};")
    h.append("")
    h.append("static const uint8_t SPECIAL_SEC[256] = {")
    h.append(fmt_u8(SPECIAL_SEC))
    h.append("};")
    h.append("")
    h.append("static const uint8_t SPECIAL_SEC_N[256] = {")
    h.append(fmt_u8(SPECIAL_SEC_N))
    h.append("};")
    h.append("")
    # tail = content[1:] (3 or 4 bytes), zero-padded to 4
    h.append("static const uint16_t SPECIAL_TAIL[17 * 4] = {")
    h.append(fmt_u16([x for e, _ in sec_tail for x in (list(e) + [0, 0, 0])[:4]]))
    h.append("};")
    h.append("")
    h.append("static const uint8_t SPECIAL_LEN[17] = {")
    h.append(fmt_u8([len(e) + 1 for e, _ in sec_tail]))
    h.append("};")
    h.append("")
    h.append("static const uint16_t SPECIAL_ID[17] = {")
    h.append(fmt_u16([tid for _, tid in sec_tail]))
    h.append("};")
    h.append("")
    h.append(f"#define MERGE_SLOTS {MERGE_SLOTS}")
    h.append("static const uint32_t MERGE_KEY[MERGE_SLOTS] = {")
    h.append(fmt_u32(merge_key))
    h.append("};")
    h.append("")
    h.append("static const uint32_t MERGE_VAL[MERGE_SLOTS] = {")
    h.append(fmt_u32(merge_val))
    h.append("};")
    h.append("")
    h.append("static const uint64_t CL_L[8192] = {")
    h.append(fmt_u64(bits_to_words(l_bits)))
    h.append("};")
    h.append("")
    h.append("static const uint64_t CL_N[8192] = {")
    h.append(fmt_u64(bits_to_words(n_bits)))
    h.append("};")
    h.append("")
    h.append("static const uint64_t CL_WS[8192] = {")
    h.append(fmt_u64(bits_to_words(ws_bits)))
    h.append("};")
    h.append("")
    h.append(f"static const uint32_t CL_L_NB[{2 * len(l_r)}] = {{")
    h.append(fmt_u32([x for r in l_r for x in r]))
    h.append("};")
    h.append("")
    h.append(f"static const uint32_t CL_N_NB[{2 * len(n_r)}] = {{")
    h.append(fmt_u32([x for r in n_r for x in r]))
    h.append("};")
    h.append("")
    h.append("#define CL_L_NB_N " + str(len(l_r)))
    h.append("#define CL_N_NB_N " + str(len(n_r)))
    h.append("")
    h.append("static const uint32_t PIECE_OFF[VOCAB_SIZE] = {")
    h.append(fmt_u32(piece_off))
    h.append("};")
    h.append("")
    h.append("static const uint8_t PIECE_LEN[VOCAB_SIZE] = {")
    h.append(fmt_u8(piece_len))
    h.append("};")
    h.append("")
    h.append(f"static const uint8_t PIECE_BYTES[{len(piece_bytes)}] = {{")
    h.append(fmt_u8(piece_bytes))
    h.append("};")
    h.append("")
    h.append("static const uint8_t CHAR_TO_BYTE[352] = {")
    h.append(fmt_u8(CHAR_TO_BYTE))
    h.append("};")
    h.append("")
    h.append(f"#define TID_SLOTS {TID_SLOTS}")
    h.append("static const uint32_t TID_KEY[TID_SLOTS] = {")
    h.append(fmt_u32(tid_key))
    h.append("};")
    h.append("")
    h.append("static const uint32_t TID_HASH[TID_SLOTS] = {")
    h.append(fmt_u32(tid_hash))
    h.append("};")
    h.append("")
    h.append("#endif  // ODIN_TOKENIZER_TABLES_H")
    out_path.write_text("\n".join(h) + "\n")

    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")
    print(
        f"  merges={n_merges}, piece bytes={len(piece_bytes)}, "
        f"non-BMP L ranges={len(l_r)}, N ranges={len(n_r)}, checksum={checksum}"
    )


if __name__ == "__main__":
    main()
