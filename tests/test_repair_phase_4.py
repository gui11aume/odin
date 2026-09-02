"""Tests for the phase-4 repair harness (runners/repair_clusters_phase_4.py).

The LLM is a scripted fake; no GPU or server is needed. Corpus-mined shapes:
the "Peter Dr. Flury" cluster, "Dyke, Kelly Van" particle mis-splits, and the
CJK false-positive guards ("Zu, Qun", "Ping, Zu").
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

RUNNER_DIR = Path(__file__).resolve().parent.parent / "runners"
if str(RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(RUNNER_DIR))

import generate_names_core as core  # noqa: E402
import repair_clusters_phase_4 as p4  # noqa: E402

# Per-tag seeds in the correct script (phase 1's filters drop wrong-script
# values, so the fake transcriber must emit real script characters).
SEEDS: dict[str, str] = {
    "cy": "Питер Флюри",
    "gk": "Πίτερ Φλούρι",
    "ab": "بيتر فلوري",
    "cn": "彼得弗吕里",
    "jp": "ピーター・フルリ",
    "kr": "페터 플루리",
    "dv": "पीटर फ्लुरी",
    "hb": "פיטר פלוורי",
    "th": "ปีเตอร์ ฟลูรี้",
    "gg": "პიტერ ფლური",
    "am": "Պիտեր Ֆլուրի",
}


def cluster_line(cells: list[tuple[str, str]]) -> str:
    return "\t".join(f"{t}{{{v}}}" for t, v in cells)


CLEAN_CELLS: list[tuple[str, str]] = [("la", "Peter Flury")] + [(t, v) for t, v in SEEDS.items()]
TAINTED_CELLS: list[tuple[str, str]] = [
    ("la", "Peter Dr. Flury"),
    ("la", "PETER DR. FLURY"),
    ("la", "Flury, Peter Dr."),
    ("la", "P. D. Flury"),
    ("cy", "Питер д-р Флюри"),
    ("cy", "Флюри П.Д."),
    ("gk", "Πίτερ δρ Φλούρι"),
    ("ab", "بيتر د. فلوري"),
    ("cn", "彼得·弗吕里"),
    ("jp", "ピーター・フルリ"),
    ("kr", "페터 플루리"),
    ("dv", "पीटर डॉ. फ्लुरी"),
    ("hb", "פיטר ד׳ר פלוורי"),
    ("th", "ปีเตอร์ ดร. ฟลูรี้"),
    ("gg", "პიტერ დრ. ფლური"),
    ("am", "Պիտեր Դր. Ֆլուրի"),
]


def flagged_views(cells: list[tuple[str, str]], line_no: int = 1) -> list[p4.ClusterView]:
    line = cluster_line(cells)
    parsed = core.parse_cluster_line(line)
    assert parsed is not None
    la_forms = [v for t, v in parsed if t == "la"]
    return [p4.flag_cluster(line_no, line, parsed, la_forms)]


# --------------------------------------------------------------------------- #
# Stage A — detectors
# --------------------------------------------------------------------------- #
def test_detect_titles_flags_dr_positions():
    forms = ["Peter Dr. Flury", "Flury Dr.", "Flury, Peter Dr."]
    flags = p4.detect_titles(forms)
    assert any(f.detector == "D1_title" and "'Dr.' (middle)" in f.evidence for f in flags)
    assert any(f.detector == "D1_title" and "'Dr.' (trail)" in f.evidence for f in flags)


def test_detect_titles_ignores_trailing_generational_and_initials():
    assert p4.detect_titles(["Peter Flury Jr.", "Peter Flury II"]) == []
    # "M." is an initialism-shaped token (single letter + period), not a title.
    assert p4.detect_titles(["Peter M. Flury"]) == []


def test_detect_generational_misplaced():
    assert any(f.detector == "D2_generational" for f in p4.detect_generational_misplaced(["II John Smith"]))
    assert p4.detect_generational_misplaced(["John Smith II"]) == []


def test_detect_particles_western_misplits():
    views = flagged_views([("la", "Dyke, Kelly Van"), ("la", "Kelly Van Dyke"), ("cn", "凯利·范·戴克")])
    kinds = {f.detector for f in views[0].flags}
    assert "D3_particle_given" in kinds
    # Lone-particle family side.
    views = flagged_views([("la", "Van, Lai Fun"), ("cn", "赖芬·万")])
    assert "D3_particle_family" in {f.detector for f in views[0].flags}


def test_detect_particles_skips_initials():
    # "Y." is a single-letter dotted initial, not the particle "y".
    assert p4.detect_particles([("la", "Flury, Y."), ("la", "Flury, Y. P.")], ["Flury, Y.", "Flury, Y. P."]) == []
    # Lone-initial family side is not a lone particle either.
    assert p4.detect_particles([("la", "Y., Peter")], ["Y., Peter"]) == []


def test_detect_particles_cjk_guard():
    # "Zu, Qun" / "Ping, Zu" are CJK names, not particle mis-splits.
    assert p4.detect_particles([("la", "Zu, Qun"), ("cn", "朱群")], ["Zu, Qun"]) == []
    assert p4.detect_particles([("la", "Ping, Zu"), ("cn", "朱平")], ["Ping, Zu"]) == []
    # Multi-token given side is still routed (the LLM adjudicates).
    assert p4.detect_particles([("la", "Lee, Young Du"), ("cn", "李庸都")], ["Lee, Young Du"]) != []


def test_detect_noise():
    flags = p4.detect_noise(["John 12 Main Street", "John Smith"])
    assert any(f.detector == "D4_digit" for f in flags)
    assert any(f.detector == "D4_noise" and "'Street'" in f.evidence for f in flags)
    assert any(f.detector == "D4_noise" and "'Corp'" in f.evidence for f in p4.detect_noise(["John Acme Corp"]))
    assert p4.detect_noise(["Peter Flury"]) == []


def test_detect_structural():
    assert any(f.detector == "D5_edge_comma" for f in p4.detect_structural(["Flury, Peter,"]))
    assert any(f.detector == "D5_multi_comma" for f in p4.detect_structural(["Flury, Peter, Jr."]))
    assert any(f.detector == "D5_duplicate" for f in p4.detect_structural(["Smith Smith"]))
    # Repeated initials are two middle initials, not a duplicate.
    assert p4.detect_structural(["John A. A. Smith"]) == []
    assert p4.detect_structural(["Peter Flury"]) == []


def test_detect_noise_park_is_a_surname():
    assert p4.detect_noise(["Sung Park", "Park, Sung"]) == []


def test_detect_nonlatin_taint():
    cells = [("cy", "Питер д-р Флюри"), ("gk", "Πίτερ δρ Φλούρι"), ("jp", "ピーター Dr フルリ")]
    flags = p4.detect_nonlatin_taint(cells)
    assert any(f.detector == "D6_title_taint" and "cy" in f.evidence for f in flags)
    assert any(f.detector == "D6_title_taint" and "gk" in f.evidence for f in flags)
    assert any(f.detector == "D6_latin_leak" and "jp" in f.evidence for f in flags)
    # Legitimate Cyrillic words that merely contain the letters д/р are not markers.
    assert p4.detect_nonlatin_taint([("cy", "Драгомир Иванов")]) == []
    # A lone Arabic initial "د." is not a title; only the word "دكتور" is.
    assert p4.detect_nonlatin_taint([("ab", "فلوري د.")]) == []
    assert any(f.detector == "D6_title_taint" for f in p4.detect_nonlatin_taint([("ab", "فلوري دكتور")]))


def test_clean_cluster_not_flagged():
    views = flagged_views(CLEAN_CELLS)
    assert views[0].flags == []
    assert views[0].given == "Peter"
    assert views[0].family == "Flury"


def test_tainted_cluster_flagged_and_recovered():
    views = flagged_views(TAINTED_CELLS)
    kinds = {f.detector for f in views[0].flags}
    assert "D1_title" in kinds
    assert "D6_title_taint" in kinds
    assert views[0].given == "Peter Dr."
    assert views[0].family == "Flury"


# --------------------------------------------------------------------------- #
# Stage B+C — end-to-end with a scripted fake LLM
# --------------------------------------------------------------------------- #
class FakeLLM:
    """Scripted stand-in for RepairLLM matching its call() contract."""

    def __init__(
        self,
        audit: list[tuple[str, dict[str, Any]]],
        fail_tags: set[str] | None = None,
        crash_after: int | None = None,
    ):
        self.audit = list(audit)
        self.fail_tags = fail_tags or set()
        self.crash_after = crash_after
        self.n_calls = 0
        self.calls: list[str] = []

    def call(self, messages, tools, **params):
        self.n_calls += 1
        if self.crash_after is not None and self.n_calls > self.crash_after:
            raise RuntimeError("simulated crash")
        user = messages[-1]["content"]
        if user.startswith("Audit one person-name cluster"):
            self.calls.append("audit")
            name, args = self.audit.pop(0) if len(self.audit) > 1 else self.audit[0]
            return name, args
        if user.startswith("Given a person name in Latin script"):
            self.calls.append("alt")
            return "emit_alt_latin", {"alts": []}
        self.calls.append("xlit")
        requested = [
            k for k in tools[0]["function"]["parameters"]["properties"] if k in SEEDS and k not in self.fail_tags
        ]
        return "emit_scripts", {tag: SEEDS[tag] for tag in requested}


def write_corpus(tmp_path: Path, lines: list[str]) -> Path:
    p = tmp_path / "corpus.txt"
    p.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return p


def run_phase4(monkeypatch, tmp_path: Path, fake: FakeLLM, lines: list[str]) -> dict[str, Any]:
    monkeypatch.setattr(p4, "RepairLLM", lambda base_url, model, seed: fake)
    inp = write_corpus(tmp_path, lines)
    out = tmp_path / "out.txt"
    report = tmp_path / "repairs.jsonl"
    meta = tmp_path / "meta.json"
    p4.main(
        [
            str(inp),
            "-o",
            str(out),
            "--report",
            str(report),
            "--meta",
            str(meta),
            "--concurrency",
            "2",
            "--wave-size",
            "10",
        ]
    )
    return {"out": out.read_text(encoding="utf-8"), "report": report, "meta": json.loads(meta.read_text())}


def test_repair_end_to_end(monkeypatch, tmp_path):
    fake = FakeLLM([("fix_name", {"given": "Peter", "family": "Flury"})])
    res = run_phase4(
        monkeypatch,
        tmp_path,
        fake,
        [cluster_line(CLEAN_CELLS), cluster_line(TAINTED_CELLS)],
    )
    lines = res["out"].splitlines()
    assert len(lines) == 2
    # Clean line is byte-identical.
    assert lines[0] == cluster_line(CLEAN_CELLS)
    # Tainted line is recomputed: no "Dr." anywhere, all twelve tags present.
    assert "Dr" not in lines[1] and "д-р" not in lines[1].lower() and "Д-Р" not in lines[1]
    cells = core.parse_cluster_line(lines[1])
    assert cells is not None
    tags = [t for t, _ in cells]
    assert tags.count("la") >= 4
    for tag in SEEDS:
        assert tag in tags
    la_forms = [v for t, v in cells if t == "la"]
    assert "Peter Flury" in la_forms
    assert all("Dr" not in v for v in la_forms)
    meta = res["meta"]
    assert meta["n_flagged"] == 1
    assert meta["n_repaired"] == 1
    assert meta["n_clean_adjudicated"] == 0
    assert meta["n_dropped"] == 0
    records = [json.loads(line) for line in res["report"].read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["outcome"] == "fixed"
    assert records[0]["given"] == "Peter Dr."
    assert records[0]["fixed_given"] == "Peter"
    assert records[0]["new_line"] == lines[1]


def test_audit_clean_adjudicated_is_byte_identical(monkeypatch, tmp_path):
    fake = FakeLLM([("name_is_clean", {})])
    res = run_phase4(monkeypatch, tmp_path, fake, [cluster_line(TAINTED_CELLS)])
    assert res["out"] == cluster_line(TAINTED_CELLS) + "\n"
    assert res["meta"]["n_clean_adjudicated"] == 1
    assert res["meta"]["n_repaired"] == 0


def test_audit_unrecoverable_drops(monkeypatch, tmp_path):
    fake = FakeLLM([("unrecoverable", {})])
    res = run_phase4(monkeypatch, tmp_path, fake, [cluster_line(CLEAN_CELLS), cluster_line(TAINTED_CELLS)])
    assert res["out"] == cluster_line(CLEAN_CELLS) + "\n"
    assert res["meta"]["n_dropped"] == 1
    assert res["meta"]["dropped_by_reason"] == {"audit_unrecoverable_or_invalid": 1}


def test_incomplete_transliteration_drops(monkeypatch, tmp_path):
    fake = FakeLLM([("fix_name", {"given": "Peter", "family": "Flury"})], fail_tags={"gg"})
    res = run_phase4(monkeypatch, tmp_path, fake, [cluster_line(TAINTED_CELLS)])
    assert res["out"] == ""
    assert res["meta"]["n_dropped"] == 1
    assert res["meta"]["dropped_by_reason"] == {"transliteration_incomplete": 1}
    record = json.loads(res["report"].read_text(encoding="utf-8"))
    assert "transliteration incomplete" in record["warnings"][0]


def test_noop_fix_is_byte_identical(monkeypatch, tmp_path):
    # fix_name echoing the original (given, family) is a no-op, not a repair.
    fake = FakeLLM([("fix_name", {"given": "Peter Dr.", "family": "Flury"})])
    res = run_phase4(monkeypatch, tmp_path, fake, [cluster_line(TAINTED_CELLS)])
    assert res["out"] == cluster_line(TAINTED_CELLS) + "\n"
    assert res["meta"]["n_clean_adjudicated"] == 1


def test_resume_after_crash(monkeypatch, tmp_path):
    # Two identical tainted lines, wave-size 1 -> two waves. Wave 1 uses five
    # calls (audit + alt + 3 xlit passes); crash on wave 2's first audit call.
    lines = [cluster_line(TAINTED_CELLS), cluster_line(TAINTED_CELLS)]
    inp = write_corpus(tmp_path, lines)
    out = tmp_path / "out.txt"
    report = tmp_path / "repairs.jsonl"
    meta = tmp_path / "meta.json"
    ckpt = tmp_path / ("out.txt.ckpt.jsonl")
    base = [str(inp), "-o", str(out), "--report", str(report), "--meta", str(meta), "--wave-size", "1"]

    fake = FakeLLM([("fix_name", {"given": "Peter", "family": "Flury"})], crash_after=5)
    monkeypatch.setattr(p4, "RepairLLM", lambda base_url, model, seed: fake)
    with pytest.raises(RuntimeError, match="simulated crash"):
        p4.main(base)
    assert ckpt.exists()
    assert out.exists() is False

    # Resume: only wave 2 remains; the repaired wave-1 line must be kept.
    fake2 = FakeLLM([("fix_name", {"given": "Peter", "family": "Flury"})])
    monkeypatch.setattr(p4, "RepairLLM", lambda base_url, model, seed: fake2)
    p4.main(base + ["--resume"])
    out_lines = out.read_text(encoding="utf-8").splitlines()
    assert len(out_lines) == 2
    assert all("Dr" not in line for line in out_lines)
    assert not ckpt.exists()
    m = json.loads(meta.read_text(encoding="utf-8"))
    assert m["n_repaired"] == 2
    assert m["n_dropped"] == 0


def test_checkpoint_without_resume_is_an_error(tmp_path):
    inp = write_corpus(tmp_path, [cluster_line(TAINTED_CELLS)])
    out = tmp_path / "out.txt"
    (tmp_path / "out.txt.ckpt.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="--resume"):
        p4.main([str(inp), "-o", str(out)])
    (tmp_path / "out.txt.ckpt.jsonl").unlink()


def test_no_llm_dry_run(tmp_path):
    inp = write_corpus(tmp_path, [cluster_line(CLEAN_CELLS), cluster_line(TAINTED_CELLS)])
    report = tmp_path / "flags.jsonl"
    meta = tmp_path / "meta.json"
    p4.main([str(inp), "--no-llm", "--report", str(report), "--meta", str(meta)])
    assert not (tmp_path / "corpus.repaired.txt").exists()
    records = [json.loads(line) for line in report.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["line_no"] == 2
    assert json.loads(meta.read_text())["n_flagged"] == 1
