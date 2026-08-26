"""Tests for the Odin VAE model."""

from __future__ import annotations

import pytest
import torch

from odin_vae.config_classes import ConfigForModel
from odin_vae.model import OdinModel

VOCAB = 64
TAGS = list(range(3, 15))  # 12 tag ids
PAD, BOS, EOS = 0, 2, 1


def make_model(decoder: str = "modernbert", device: str = "cpu") -> OdinModel:
    config = ConfigForModel(
        tokenizer_path="unused",
        hidden_size=32,
        attention_heads=2,
        intermediate_size=64,
        encoder_layers=2,
        decoder_layers=3,
        local_attention=16,
        max_position_embeddings=32,
        decoder=decoder,
        kl_weight=1e-3,
    )
    model = OdinModel(config, vocab_size=VOCAB, pad_token_id=PAD, bos_token_id=BOS, eos_token_id=EOS)
    return model.to(device)


def make_batch(b: int = 4, k_in: int = 3, k_out: int = 2, l_in: int = 6, l_out: int = 4, device: str = "cpu") -> dict:
    torch.manual_seed(0)
    surf_ids = torch.randint(3, VOCAB, (b * k_in, l_in), device=device)
    surf_mask = torch.ones(b * k_in, l_in, dtype=torch.long, device=device)
    surf_mask[0, -2:] = 0
    surf_mask[1, -1:] = 0
    target_ids = torch.randint(3, VOCAB, (b * k_out, l_out), device=device)
    target_mask = torch.ones(b * k_out, l_out, dtype=torch.long, device=device)
    target_mask[0, -1:] = 0
    target_tags = torch.tensor(TAGS[: b * k_out], device=device)
    return {
        "surf_ids": surf_ids,
        "surf_mask": surf_mask,
        "target_ids": target_ids,
        "target_mask": target_mask,
        "target_tags": target_tags,
        "n_clusters": b,
    }


@pytest.mark.parametrize("decoder", ["modernbert", "t5"])
def test_forward_loss_finite_and_backward(decoder: str) -> None:
    model = make_model(decoder)
    out = model(**make_batch())
    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["ce"])
    assert torch.isfinite(out["kl"])
    out["loss"].backward()
    assert model.pool_query.grad is not None
    assert model.mu_head.weight.grad is not None
    assert model.logvar_head.weight.grad is not None
    if decoder == "modernbert":
        assert model.z_proj.weight.grad is not None
    assert model.encoder.parameters().__next__().grad is not None


def test_embeddings_tied() -> None:
    model = make_model("modernbert")
    shared = model.decoder.embeddings.tok_embeddings.weight
    assert model.encoder.embeddings.tok_embeddings.weight is shared
    assert model.lm_head.weight is shared
    model_t5 = make_model("t5")
    shared_t5 = model_t5.decoder.embed_tokens.weight
    assert model_t5.encoder.embeddings.tok_embeddings.weight is shared_t5
    assert model_t5.lm_head.weight is shared_t5


def test_permutation_invariance_of_mu() -> None:
    model = make_model("modernbert").eval()
    batch = make_batch(b=2, k_in=4, k_out=1)
    with torch.no_grad():
        surf_ids = batch["surf_ids"]
        surf_mask = batch["surf_mask"]
        # Cluster 0: surfaces [0,1,2,3]; cluster 1: the same surfaces shuffled.
        shuffled = torch.cat([surf_ids[[3, 0, 1, 2]]])
        mask_sh = torch.cat([surf_mask[[3, 0, 1, 2]]])
        both_ids = torch.cat([surf_ids[:4], shuffled])
        both_mask = torch.cat([surf_mask[:4], mask_sh])
        mu, _ = model.encode(both_ids, both_mask, k=4)
    assert mu.shape == (2, model.config.hidden_size)
    assert torch.allclose(mu[0], mu[1], atol=1e-5)


def test_z_influences_logits() -> None:
    model = make_model("modernbert").eval()
    batch = make_batch(b=1, k_in=2, k_out=1)
    with torch.no_grad():
        mu, _ = model.encode(batch["surf_ids"], batch["surf_mask"], k=2)
        z_a = model.z_proj(mu)
        z_b = model.z_proj(mu + 2.0)
        tags_col = batch["target_tags"].unsqueeze(1)
        din = torch.cat([torch.full((1, 1), BOS, dtype=torch.long), tags_col, batch["target_ids"][:, :-1]], dim=1)
        emb_a = torch.cat([z_a.unsqueeze(1), model.decoder.embeddings(din)], dim=1)
        emb_b = torch.cat([z_b.unsqueeze(1), model.decoder.embeddings(din)], dim=1)
        pos = torch.arange(emb_a.shape[1]).unsqueeze(0)
        la = model.lm_head(model.decoder(inputs_embeds=emb_a, position_ids=pos).last_hidden_state)
        lb = model.lm_head(model.decoder(inputs_embeds=emb_b, position_ids=pos).last_hidden_state)
    assert (la - lb).abs().max() > 1e-3


def test_generate_stops_and_respects_cap() -> None:
    for decoder in ("modernbert", "t5"):
        model = make_model(decoder).eval()
        z = torch.zeros(model.config.hidden_size)
        # Untrained model: cap must bound the length.
        ids = model.generate(z, TAGS[0], max_new_tokens=5)
        assert len(ids) <= 5
        assert all(PAD <= t < VOCAB for t in ids)
        # Greedy decoding is deterministic.
        ids2 = model.generate(z, TAGS[0], max_new_tokens=5)
        assert ids == ids2


def test_generate_sampling_with_seed_reproducible() -> None:
    model = make_model("modernbert").eval()
    z = torch.randn(model.config.hidden_size)
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    ids1 = model.generate(z, TAGS[1], max_new_tokens=4, temperature=1.0, top_p=0.9, generator=g1)
    ids2 = model.generate(z, TAGS[1], max_new_tokens=4, temperature=1.0, top_p=0.9, generator=g2)
    assert ids1 == ids2


def test_encode_k_validation() -> None:
    model = make_model("modernbert")
    ids = torch.randint(3, VOCAB, (4, 5))
    mask = torch.ones(4, 5, dtype=torch.long)
    with torch.no_grad():
        mu, _ = model.encode(ids, mask)  # one cluster of 4
        assert mu.shape == (1, model.config.hidden_size)
        mu, _ = model.encode(ids, mask, k=2)
        assert mu.shape == (2, model.config.hidden_size)
    with pytest.raises(ValueError, match="multiple"):
        model.encode(ids, mask, k=3)


def test_kl_weight_zero_drops_kl() -> None:
    config = ConfigForModel(
        tokenizer_path="unused",
        hidden_size=32,
        attention_heads=2,
        intermediate_size=64,
        encoder_layers=2,
        decoder_layers=3,
        local_attention=16,
        max_position_embeddings=32,
        decoder="modernbert",
        kl_weight=0.0,
    )
    model = OdinModel(config, vocab_size=VOCAB, pad_token_id=PAD, bos_token_id=BOS, eos_token_id=EOS)
    out = model(**make_batch())
    assert torch.isfinite(out["loss"])
    assert out["loss"].item() == pytest.approx(out["ce"].item(), rel=1e-5)
