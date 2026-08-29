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


def make_batch(
    b: int = 4,
    k_in: int = 3,
    k_out: int = 2,
    l_in: int = 6,
    l_out: int = 4,
    primed: bool = True,
    k_per_cluster: list[int] | None = None,
    target_primed: list[int] | None = None,
    device: str = "cpu",
) -> dict:
    """A model batch. ``primed=True`` keeps the classic all-primed targets."""
    torch.manual_seed(0)
    if k_per_cluster is None:
        k_per_cluster = [k_in] * b
    n_in = sum(k_per_cluster)
    surf_ids = torch.randint(3, VOCAB, (n_in, l_in), device=device)
    surf_mask = torch.ones(n_in, l_in, dtype=torch.long, device=device)
    if n_in >= 2:
        surf_mask[0, -2:] = 0
        surf_mask[1, -1:] = 0
    else:
        surf_mask[0, -1:] = 0
    nk = b * k_out
    target_ids = torch.randint(3, VOCAB, (nk, l_out), device=device)
    target_mask = torch.ones(nk, l_out, dtype=torch.long, device=device)
    target_mask[0, -1:] = 0
    target_tags = torch.tensor(TAGS[:nk], device=device)
    if target_primed is None:
        target_primed = [1 if primed else 0] * nk
    return {
        "surf_ids": surf_ids,
        "surf_mask": surf_mask,
        "k_per_cluster": torch.tensor(k_per_cluster, dtype=torch.long, device=device),
        "target_ids": target_ids,
        "target_mask": target_mask,
        "target_tags": target_tags,
        "target_primed": torch.tensor(target_primed, dtype=torch.long, device=device),
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


def test_forward_mixed_priming_and_variable_k() -> None:
    for decoder in ("modernbert", "t5"):
        model = make_model(decoder)
        # Cluster sizes 1..4 (variable-k encoder path) and a mix of
        # primed/unprimed targets in the same batch.
        k = [1, 2, 3, 4]
        nk = 4 * 2
        batch = make_batch(b=4, k_per_cluster=k, k_out=2, target_primed=[i % 2 for i in range(nk)])
        out = model(**batch)
        assert torch.isfinite(out["loss"])
        out["loss"].backward()
        assert model.pool_query.grad is not None


def test_priming_changes_decoder_input() -> None:
    """Primed rows reach the decoder as [BOS, tag, ...]; unprimed as [BOS, ...]."""
    model = make_model("modernbert").eval()
    batch = make_batch(b=1, k_in=1, k_out=2, target_primed=[1, 0])
    embed = model._decoder_embed_module()
    captured: list[torch.Tensor] = []

    class Spy(torch.nn.Module):
        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            captured.append(x)
            return embed(x)

    model.decoder.embeddings.tok_embeddings = Spy()
    with torch.no_grad():
        model(**batch)
    dec = captured[0]
    tag = batch["target_tags"][0]
    assert dec[0, 0] == BOS and dec[0, 1] == tag  # primed row: tag at position 1
    assert (dec[0, 2:] == batch["target_ids"][0, :-1]).all()  # then t0..t_{L-2}
    assert dec[1, 0] == BOS and dec[1, 1] == batch["target_ids"][1, 0]  # unprimed: no tag
    assert (dec[1, 1:-1] == batch["target_ids"][1, :-1]).all()
    assert dec[1, -1] == PAD


def test_generate_unprimed() -> None:
    for decoder in ("modernbert", "t5"):
        model = make_model(decoder).eval()
        z = torch.zeros(model.config.hidden_size)
        ids = model.generate(z, None, max_new_tokens=5)
        assert len(ids) <= 5
        assert all(PAD <= t < VOCAB for t in ids)
        # Greedy decoding is deterministic, and differs from primed decoding
        # (different seed sequence) with overwhelming probability.
        assert ids == model.generate(z, None, max_new_tokens=5)
        primed_ids = model.generate(z, TAGS[0], max_new_tokens=5)
        assert ids != primed_ids


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


def _freeze_variance(model: OdinModel) -> None:
    """Make the reparameterized sample collapse to mu (std ~ 1e-9)."""
    model.logvar_head.weight.data.zero_()
    model.logvar_head.bias.data.fill_(-40.0)


@pytest.mark.parametrize("decoder", ["modernbert", "t5"])
def test_log_prob_shape_finite_deterministic(decoder: str) -> None:
    model = make_model(decoder).eval()
    z = torch.randn(model.config.hidden_size)
    ids = [5, 6, 7]
    total_p, per_p = model.log_prob(z, TAGS[0], ids)
    total_u, per_u = model.log_prob(z, None, ids)
    # Default (include_eos): complete sequence. Primed predicts the tag token
    # first, then the surface, then EOS; unprimed the surface then EOS.
    assert len(per_p) == len(ids) + 2
    assert len(per_u) == len(ids) + 1
    assert all(torch.isfinite(torch.tensor(v)) for v in per_p + per_u)
    assert total_p == pytest.approx(sum(per_p), abs=1e-6)
    # Without the stop decision the predictions are the tag? + surface only.
    assert len(model.log_prob(z, TAGS[0], ids, include_eos=False)[1]) == len(ids) + 1
    assert len(model.log_prob(z, None, ids, include_eos=False)[1]) == len(ids)
    # Deterministic, and sensitive to the input surface.
    assert model.log_prob(z, TAGS[0], ids) == (total_p, per_p)
    assert model.log_prob(z, TAGS[0], [5, 6, 8])[0] != total_p


def test_log_prob_single_token_surface() -> None:
    model = make_model().eval()
    z = torch.zeros(model.config.hidden_size)
    total_p, per_p = model.log_prob(z, TAGS[1], [9])
    total_u, per_u = model.log_prob(z, None, [9])
    assert len(per_p) == 3  # tag, token, EOS
    assert len(per_u) == 2  # token, EOS
    assert all(torch.isfinite(torch.tensor(v)) for v in per_p + per_u)


@pytest.mark.parametrize("primed", [True, False])
@pytest.mark.parametrize("decoder", ["modernbert", "t5"])
def test_log_prob_matches_forward_ce(decoder: str, primed: bool) -> None:
    """With the variance frozen, forward's CE on a single target row must equal
    the mean per-token NLL returned by log_prob (teacher-forcing alignment)."""
    model = make_model(decoder).eval()  # eval: dropout off, so forward is deterministic
    _freeze_variance(model)
    torch.manual_seed(7)
    ids = [4, 11, 19, 25]
    n = len(ids)
    l_out = n + 3  # surface + EOS + padding tail (the collator appends EOS)
    batch = make_batch(b=1, k_in=2, k_out=1, l_in=5, l_out=l_out, primed=primed)
    batch["target_ids"] = torch.tensor([[*ids, EOS, PAD]], dtype=torch.long)
    batch["target_mask"] = torch.tensor([[1] * (n + 1) + [0]], dtype=torch.long)
    batch["target_tags"] = torch.tensor([TAGS[3]], dtype=torch.long)
    batch["target_primed"] = torch.tensor([1 if primed else 0], dtype=torch.long)
    out = model(**batch)
    mu, _ = model.encode(batch["surf_ids"], batch["surf_mask"], k_per_cluster=batch["k_per_cluster"])
    tag_id = TAGS[3] if primed else None
    total, per = model.log_prob(mu[0], tag_id, ids)  # include_eos=True, like training
    assert out["ce"].item() == pytest.approx(-sum(per) / len(per), abs=1e-4)
    # The primed prediction includes the tag token; both end in the EOS term.
    assert len(per) == n + 1 + (1 if primed else 0)


@pytest.fixture()
def cluster_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(3)
    ids = torch.randint(3, VOCAB, (3, 6))
    mask = torch.ones(3, 6, dtype=torch.long)
    mask[0, -1:] = 0
    mask[2, -2:] = 0
    return ids, mask


@pytest.mark.parametrize("decoder", ["modernbert", "t5"])
def test_sample_generate_shapes_and_seed_determinism(
    decoder: str, cluster_inputs: tuple[torch.Tensor, torch.Tensor]
) -> None:
    model = make_model(decoder).eval()
    ids, mask = cluster_inputs
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    s1, mu, logvar = model.sample_generate(
        ids, mask, k=3, tag_id=TAGS[0], n_samples=4, max_new_tokens=6, temperature=1.0, generator=g1
    )
    s2, mu2, logvar2 = model.sample_generate(
        ids, mask, k=3, tag_id=TAGS[0], n_samples=4, max_new_tokens=6, temperature=1.0, generator=g2
    )
    assert len(s1) == 4
    for sample in s1:
        assert len(sample) <= 6
        assert all(PAD <= t < VOCAB for t in sample)
    assert mu.shape == (1, model.config.hidden_size)
    assert logvar.shape == (1, model.config.hidden_size)
    assert torch.isfinite(mu).all() and torch.isfinite(logvar).all()
    # Same seed -> same latent draws -> same decodes (incl. token sampling).
    assert s1 == s2
    assert torch.equal(mu, mu2) and torch.equal(logvar, logvar2)
    # The posterior equals a direct encode of the same cluster.
    with torch.no_grad():
        ref_mu, ref_logvar = model.encode(ids, mask, k=3)
    assert torch.equal(mu, ref_mu) and torch.equal(logvar, ref_logvar)


def test_sample_generate_frozen_variance_collapses_to_mu(cluster_inputs: tuple[torch.Tensor, torch.Tensor]) -> None:
    """With std ~ 0 every sample is z == mu, hence all decodes are identical."""
    model = make_model().eval()
    _freeze_variance(model)
    ids, mask = cluster_inputs
    s, mu, _ = model.sample_generate(ids, mask, k=3, tag_id=TAGS[1], n_samples=5, max_new_tokens=6)
    assert len(set(map(tuple, s))) == 1
    # All samples are z == mu, i.e. the plain mu decode.
    assert s[0] == model.generate(mu[0], TAGS[1], max_new_tokens=6)


def test_sample_generate_validation(cluster_inputs: tuple[torch.Tensor, torch.Tensor]) -> None:
    model = make_model()
    ids, mask = cluster_inputs
    with pytest.raises(ValueError, match="n_samples"):
        model.sample_generate(ids, mask, k=3, n_samples=0)
    # k=1 over 3 rows is three clusters; sampling decodes one cluster at a time.
    with pytest.raises(ValueError, match="one cluster"):
        model.sample_generate(ids, mask, k=1, n_samples=1)
