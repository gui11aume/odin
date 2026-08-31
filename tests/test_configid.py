"""Tests for the two-part architecture checksum (configid)."""

from __future__ import annotations

import re

import pytest
import torch

from odin_vae.config_classes import ConfigForModel
from odin_vae.configid import (
    ARCH_FIELDS,
    SPEC_VERSION,
    configid,
    configid_of_checkpoint,
    layout_id,
    layout_spec,
    shape_map_of,
    verify_configid,
    wiring_id,
    wiring_spec,
)
from odin_vae.model import OdinModel

CODE_RE = re.compile(r"^[0-9A-V]{3}\.[0-9A-V]{3}$")
LEGACY_RE = re.compile(r"^[A-Z2-7]{4}$")
CROCKFORD = set("0123456789ABCDEFGHJKMNPQRSTVWXYZ")


def small_cfg(**overrides) -> ConfigForModel:
    base = dict(
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
    base.update(overrides)
    return ConfigForModel(**base)


def small_model(vocab_size: int = 64, **overrides) -> OdinModel:
    return OdinModel(
        small_cfg(**overrides),
        vocab_size=vocab_size,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


@pytest.fixture(scope="module")
def model() -> OdinModel:
    return small_model()


def test_format(model):
    assert CODE_RE.fullmatch(model.configid)
    assert set(model.configid) <= CROCKFORD | {"."}


def test_not_confusable_with_legacy_code(model):
    # Legacy bare 4-char standard-base32 codes are a single chunk, no dot.
    assert not LEGACY_RE.fullmatch(model.configid)


def test_specs_start_with_version(model):
    assert wiring_spec(model.config).startswith(SPEC_VERSION + "\n")
    assert layout_spec(shape_map_of(model)).startswith(SPEC_VERSION + "\n")


def test_deterministic_and_order_invariant(model):
    smap = shape_map_of(model)
    shuffled = dict(reversed(list(smap.items())))
    assert layout_id(smap) == layout_id(shuffled)
    assert model.configid == configid(model.config, model)


def test_wiring_field_sensitivity():
    base = small_cfg()
    mutations = {
        "hidden_size": 64,
        "attention_heads": 4,
        "intermediate_size": 128,
        "encoder_layers": 3,
        "decoder_layers": 3,
        "local_attention": 32,
        "max_position_embeddings": 64,
        "decoder": "t5",
    }
    assert set(mutations) == set(ARCH_FIELDS)
    for field, value in mutations.items():
        assert wiring_id(small_cfg(**{field: value})) != wiring_id(base), field


def test_wiring_excludes_training_fields():
    base = small_cfg()
    for overrides in (
        {"kl_weight": 0.5},
        {"num_latent_samples": 8},
        {"use_fast_tokenizer": False},
        {"tokenizer_path": "elsewhere"},
    ):
        assert wiring_id(small_cfg(**overrides)) == wiring_id(base), overrides


def test_vocab_change_flips_layout_only():
    m1, m2 = small_model(vocab_size=64), small_model(vocab_size=128)
    w1, l1 = m1.configid.split(".")
    w2, l2 = m2.configid.split(".")
    assert w1 == w2
    assert l1 != l2


def test_wiring_only_change_flips_wiring_only():
    m1, m2 = small_model(), small_model(local_attention=32)
    w1, l1 = m1.configid.split(".")
    w2, l2 = m2.configid.split(".")
    assert w1 != w2
    assert l1 == l2


def test_layout_backstop_catches_extra_tensor(model):
    smap = shape_map_of(model)
    smap["extra.head.weight"] = (32, 32)
    assert layout_id(smap) != layout_id(shape_map_of(model))


def test_checkpoint_roundtrip(model, tmp_path):
    ckpt = tmp_path / "m.ckpt"
    state = {f"model.{k}": v for k, v in model.state_dict().items()}
    torch.save({"state_dict": state}, ckpt)
    assert configid_of_checkpoint(model.config, ckpt) == model.configid
    assert verify_configid(model, ckpt) == model.configid


def test_checkpoint_roundtrip_with_stamp(model, tmp_path):
    ckpt = tmp_path / "m.ckpt"
    state = {f"model.{k}": v for k, v in model.state_dict().items()}
    torch.save({"state_dict": state, "hyper_parameters": {"configid": model.configid}}, ckpt)
    assert verify_configid(model, ckpt) == model.configid


def test_verify_rejects_layout_mismatch(model, tmp_path):
    other = small_model(vocab_size=96)
    ckpt = tmp_path / "other.ckpt"
    torch.save({"state_dict": dict(other.state_dict())}, ckpt)
    with pytest.raises(ValueError, match="layout mismatch"):
        verify_configid(model, ckpt)


def test_verify_rejects_wiring_mismatch(model, tmp_path):
    other_wiring = wiring_id(small_cfg(local_attention=32))
    stored = f"{other_wiring}.{layout_id(shape_map_of(model))}"
    ckpt = tmp_path / "other.ckpt"
    torch.save(
        {"state_dict": dict(model.state_dict()), "hyper_parameters": {"configid": stored}},
        ckpt,
    )
    with pytest.raises(ValueError, match="wiring mismatch"):
        verify_configid(model, ckpt)
