"""Tests for the Odin VAE inference API."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import PreTrainedTokenizerFast

from odin_vae.config_classes import ConfigForModel, ConfigForRoot
from odin_vae.inference import OdinInference
from odin_vae.model import OdinModel

REAL_TOKENIZER = Path("/mnt/nvme1/odin_tokenizer")

SURFACES = [("la", "J. Smith"), ("la", "J. Smith"), ("gk", "Иван Иванов")]
SINGLE = [("la", "J. Smith")]


@pytest.fixture(scope="session")
def tokenizer():
    if not REAL_TOKENIZER.is_dir():
        pytest.skip("real tokenizer directory not mounted")
    return PreTrainedTokenizerFast.from_pretrained(str(REAL_TOKENIZER))  # nosec: B615  # local directory, not the Hub


@pytest.fixture(scope="session")
def small_model(tokenizer):
    torch.manual_seed(0)
    config = ConfigForModel(
        tokenizer_path="unused",
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
        config,
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return model


@pytest.fixture()
def odin(small_model, tokenizer):
    return OdinInference(small_model, tokenizer, device="cpu")


def test_posterior_is_deterministic(odin) -> None:
    a = odin.posterior(SINGLE)
    b = odin.posterior(SINGLE)
    assert a.mu.shape == a.logvar.shape
    assert torch.allclose(a.mu, b.mu)
    assert torch.allclose(a.logvar, b.logvar)


def test_posterior_kl_and_variance_sane(odin) -> None:
    post = odin.posterior(SINGLE)
    assert post.kl_to_prior() > 0.0
    assert post.mean_variance() > 0.0
    assert (post.std > 0).all()


def test_encoder_rows_follow_harness_protocol(odin) -> None:
    rows = odin._encoder_rows(SURFACES)
    surf_ids, surf_mask = rows
    assert surf_ids.shape == surf_mask.shape
    assert surf_ids[0, 0] == odin.tag_ids["la"]
    assert surf_ids[2, 0] == odin.tag_ids["gk"]
    # Tag + first surface token are real; padding is pad id with mask 0.
    assert surf_mask[0, 0] == 1
    length = int(surf_mask[0].sum())
    assert surf_ids[0, length:].eq(odin.model.pad_token_id).all()
    assert surf_mask[0, length:].eq(0).all()
    # Row 1 is identical to row 0 (same tag + text).
    assert torch.equal(surf_ids[0], surf_ids[1])
    assert torch.equal(surf_mask[0], surf_mask[1])


def test_max_surfaces_caps_rows(odin) -> None:
    rows = odin._encoder_rows([("la", f"name {i}") for i in range(20)])
    assert rows[0].shape[0] == odin.max_surfaces


def test_score_surface_matches_log_prob(odin) -> None:
    mu = odin.posterior(SINGLE).mu
    tag = "la"
    text = "John Allan Smith"
    ids = odin.tokenizer.encode(text, add_special_tokens=False)
    lp, per = odin.model.log_prob(mu, odin.tag_ids[tag], ids)
    assert odin.score_surface(mu, text, tag) == pytest.approx(-lp / len(per))
    lp_un, per_un = odin.model.log_prob(mu, None, ids)
    assert odin.score_surface(mu, text, tag, primed=False) == pytest.approx(-lp_un / len(per_un))


def test_score_cluster_is_surface_mean(odin) -> None:
    mu = odin.posterior(SINGLE).mu
    surfaces = [("la", "John Allan Smith"), ("la", "J. A. Smith")]
    expected = sum(odin.score_surface(mu, t, tag) for tag, t in surfaces) / len(surfaces)
    assert odin.score_cluster(mu, surfaces) == pytest.approx(expected)


def test_match_sorts_and_margins(odin) -> None:
    candidates = {
        "A": [("la", "John Allan Smith")],
        "B": [("la", "Acme Corp")],
        "C": [("la", "Smith, J.")],
    }
    mu = odin.posterior(SINGLE).mu
    results = odin.match(SINGLE, candidates)
    assert [r.name for r in results] == sorted(candidates, key=lambda n: odin.score_cluster(mu, candidates[n]))
    scores = [r.score for r in results]
    assert scores == sorted(scores)
    for i, r in enumerate(results):
        others = scores[:i] + scores[i + 1 :]
        median = (
            others[len(others) // 2]
            if len(others) % 2
            else 0.5 * (others[len(others) // 2 - 1] + others[len(others) // 2])
        )
        assert r.margin == pytest.approx(median - r.score)


def test_generate_is_reproducible_with_seed(odin) -> None:
    a = odin.generate(SINGLE, "la", n_samples=3, temperature=0.7, seed=7)
    b = odin.generate(SINGLE, "la", n_samples=3, temperature=0.7, seed=7)
    assert a == b
    assert len(a) == 3
    assert all(isinstance(s, str) for s in a)


def test_generate_greedy_mode_reproducible(odin) -> None:
    a = odin.generate(SINGLE, "la", n_samples=2, temperature=0.0, seed=7)
    b = odin.generate(SINGLE, "la", n_samples=2, temperature=0.0, seed=7)
    assert a == b


def test_decode_matches_direct_greedy(odin) -> None:
    mu = odin.posterior(SINGLE).mu
    ids = odin.model.generate(mu, odin.tag_ids["la"], max_new_tokens=48)
    expected = odin.tokenizer.decode(ids, skip_special_tokens=True)
    assert odin.decode(SINGLE, "la") == expected


def test_decode_returns_string(odin) -> None:
    out = odin.decode(SINGLE, "la")
    assert isinstance(out, str)


def test_unknown_tag_rejected(odin) -> None:
    with pytest.raises(ValueError, match="Unknown script tag"):
        odin.posterior([("xx", "Nope")])
    with pytest.raises(ValueError, match="Unknown script tag"):
        odin.score_surface(torch.zeros(32), "text", "xx")


def test_plain_strings_require_shared_tag(odin) -> None:
    with pytest.raises(ValueError, match="shared `tag`"):
        odin.posterior(["J. Smith"])
    assert odin.posterior(["J. Smith"], "la").mu.shape[0] > 0


def test_from_checkpoint_roundtrip(odin, small_model, tokenizer, tmp_path) -> None:
    state = {f"model.{k}": v for k, v in small_model.state_dict().items()}
    ckpt_path = tmp_path / "009.ckpt"
    torch.save({"state_dict": state}, str(ckpt_path))
    config = ConfigForRoot.from_mapping(
        {
            "data_root": "unused",
            "splits": {"train": {"dataset": {"pattern": "unused"}}},
            "model": {
                "tokenizer_path": tokenizer.name_or_path,
                "hidden_size": 32,
                "attention_heads": 2,
                "intermediate_size": 64,
                "encoder_layers": 2,
                "decoder_layers": 2,
                "local_attention": 16,
                "max_position_embeddings": 32,
            },
        }
    )
    loaded = OdinInference.from_checkpoint(ckpt_path, config, device="cpu")
    a = odin.posterior(SINGLE).mu
    b = loaded.posterior(SINGLE).mu
    assert torch.allclose(a, b, atol=1e-5)
