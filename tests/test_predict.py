"""Tests for the predict harness (runners/predict.py)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
import torch
import yaml
from transformers import PreTrainedTokenizerFast

from odin_vae.config_classes import ConfigForModel, ConfigForRoot
from odin_vae.inference import OdinInference
from odin_vae.model import OdinModel

RUNNER_DIR = Path(__file__).resolve().parent.parent / "runners"
if str(RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(RUNNER_DIR))

import predict  # noqa: E402

REAL_TOKENIZER = Path("/mnt/nvme1/odin_tokenizer")
CONFIGID_RE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{3}\.[0-9A-HJKMNP-TV-Z]{3}$")

QUERIES = [
    {"id": "q1", "surfaces": [{"tag": "la", "text": "J. Smith"}, {"tag": "gk", "text": "Ιωάννης Σμιθ"}]},
    {"id": "q2", "surfaces": ["Acme"], "tag": "la"},
    {"id": "empty", "surfaces": []},
]
PANEL = [
    {"name": "acme", "surfaces": [{"tag": "la", "text": "Acme"}, {"tag": "la", "text": "ACME CORP"}]},
    {"name": "smith", "surfaces": [{"tag": "la", "text": "John Smith"}]},
    {"name": "flury", "surfaces": [{"tag": "la", "text": "Peter Flury"}]},
]


def write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def env(tmp_path_factory):
    if not REAL_TOKENIZER.is_dir():
        pytest.skip("real tokenizer directory not mounted")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(REAL_TOKENIZER))  # nosec: B615  # local directory, not the Hub
    torch.manual_seed(0)
    cfg = ConfigForModel(
        tokenizer_path=str(REAL_TOKENIZER),
        hidden_size=32,
        attention_heads=2,
        intermediate_size=64,
        encoder_layers=2,
        decoder_layers=2,
        local_attention=16,
        max_position_embeddings=32,
        decoder="modernbert",
    )
    model = OdinModel(
        cfg,
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    d = tmp_path_factory.mktemp("predict_env")
    ckpt = d / "m.ckpt"
    torch.save({"state_dict": {f"model.{k}": v for k, v in model.state_dict().items()}}, ckpt)
    root_cfg = ConfigForRoot.from_mapping(
        {
            "data_root": str(d / "unused"),
            "splits": {"train": {"dataset": {"pattern": "train/shard-{000000..000000}.tar.gz"}}},
            "model": cfg.model_dump(),
        }
    )
    config_yaml = d / "config.yaml"
    config_yaml.write_text(yaml.safe_dump(root_cfg.model_dump()), encoding="utf-8")
    odin = OdinInference.from_checkpoint(ckpt, root_cfg, device="cpu")
    return {"odin": odin, "ckpt": ckpt, "config": config_yaml}


def test_load_clusters(env, tmp_path):
    clusters, skipped = predict.load_clusters(write_jsonl(tmp_path / "queries.jsonl", QUERIES), key_field="id")
    assert skipped == 1
    assert [c.key for c in clusters] == ["q1", "q2"]
    assert clusters[0].rows == (("la", "J. Smith"), ("gk", "Ιωάννης Σμιθ"))
    assert clusters[1].rows == (("la", "Acme"),)


def test_load_clusters_rejects_unknown_tag(tmp_path):
    bad = [
        {"id": "ok", "surfaces": [{"tag": "la", "text": "A"}]},
        {"id": "bad", "surfaces": [{"tag": "xx", "text": "B"}]},
    ]
    with pytest.raises(ValueError, match="line 2: unknown script tag"):
        predict.load_clusters(write_jsonl(tmp_path / "bad.jsonl", bad), key_field="id")


def test_embed_schema_and_determinism(env, tmp_path):
    odin = env["odin"]
    clusters, _ = predict.load_clusters(write_jsonl(tmp_path / "queries.jsonl", QUERIES), key_field="id")
    rows = predict.embed_clusters(odin, clusters)
    assert [r["id"] for r in rows] == ["q1", "q2"]
    for r in rows:
        assert len(r["mu"]) == 32
        assert len(r["logvar"]) == 32
        for field in ("kl_to_prior", "mean_variance", "median_std", "max_variance", "dims_over_prior"):
            assert isinstance(r[field], float)
    assert rows == predict.embed_clusters(odin, clusters)


def test_embed_is_order_invariant(env, tmp_path):
    odin = env["odin"]
    clusters, _ = predict.load_clusters(write_jsonl(tmp_path / "queries.jsonl", QUERIES), key_field="id")
    forward = predict.embed_clusters(odin, clusters)
    backward = predict.embed_clusters(odin, list(reversed(clusters)))
    assert {r["id"]: r["mu"] for r in backward} == {r["id"]: r["mu"] for r in forward}


def test_variants_schema_and_determinism(env, tmp_path):
    odin = env["odin"]
    clusters, _ = predict.load_clusters(write_jsonl(tmp_path / "queries.jsonl", QUERIES), key_field="id")
    kwargs = dict(tags=["la"], n_samples=3, max_new_tokens=24, temperature=0.0, top_p=None, seed=7)
    rows = predict.generate_variants(odin, clusters, **kwargs)
    assert [(r["id"], r["tag"]) for r in rows] == [("q1", "la"), ("q2", "la")]
    for r in rows:
        # The untrained fixture model may decode straight to EOS, so only the
        # types are asserted here (content is covered by the real-ckpt smoke run).
        assert isinstance(r["greedy"], str)
        assert len(r["variants"]) == 3
        assert all(isinstance(v, str) for v in r["variants"])
    assert rows == predict.generate_variants(odin, clusters, **kwargs)


def test_score_cross_product(env, tmp_path):
    odin = env["odin"]
    queries, _ = predict.load_clusters(write_jsonl(tmp_path / "queries.jsonl", QUERIES), key_field="id")
    panel, _ = predict.load_clusters(write_jsonl(tmp_path / "panel.jsonl", PANEL), key_field="name")
    rows = predict.score_panel(odin, queries, panel, priming="both")
    assert len(rows) == 2 * 3 * 2
    assert {(r["id"], r["candidate"], r["primed"]) for r in rows} == {
        (q, p, f) for q in ("q1", "q2") for p in ("acme", "smith", "flury") for f in (True, False)
    }
    assert all(r["nll_tok"] > 0 for r in rows)


def test_match_agrees_with_api(env, tmp_path):
    odin = env["odin"]
    query, _ = predict.load_clusters(write_jsonl(tmp_path / "query.jsonl", QUERIES[:1]), key_field="id")
    panel, _ = predict.load_clusters(write_jsonl(tmp_path / "panel.jsonl", PANEL), key_field="name")
    rows = predict.match_panel(odin, query, panel, priming="primed", top=2)
    assert len(rows) == 1
    assert len(rows[0]["ranked"]) == 2
    assert rows[0]["top_name"] == rows[0]["ranked"][0]["name"]
    direct = odin.match(list(query[0].rows), {m.key: list(m.rows) for m in panel}, primed=True)
    assert [r["name"] for r in rows[0]["ranked"]] == [d.name for d in direct][:2]
    assert [r["score"] for r in rows[0]["ranked"]] == [d.score for d in direct][:2]


def test_main_embed_end_to_end(env, tmp_path):
    odin = env["odin"]
    queries_path = write_jsonl(tmp_path / "queries.jsonl", QUERIES)
    out = tmp_path / "out" / "embed.jsonl"
    predict.main(
        [
            "--config",
            str(env["config"]),
            "--checkpoint",
            str(env["ckpt"]),
            "--mode",
            "embed",
            "--input",
            str(queries_path),
            "--out",
            str(out),
            "--device",
            "cpu",
        ]
    )
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in rows] == ["q1", "q2"]
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert CONFIGID_RE.fullmatch(meta["configid"])
    assert meta["configid"] == odin.model.configid
    assert meta["n_skipped_empty"] == 1
    assert meta["mode"] == "embed"


def test_main_requires_panel_for_match(env, tmp_path):
    queries_path = write_jsonl(tmp_path / "queries.jsonl", QUERIES[:1])
    with pytest.raises(SystemExit, match="--panel is required"):
        predict.main(
            [
                "--config",
                str(env["config"]),
                "--checkpoint",
                str(env["ckpt"]),
                "--mode",
                "match",
                "--input",
                str(queries_path),
            ]
        )
