"""Odin VAE model: permutation-invariant encoder + tag-primed decoder.

Architecture (encoder, latent and decoder all share ``hidden_size``):

* Encoder: ``ModernBertModel``. Each surface (``[tag] + tokens``) is encoded
  independently; the last non-pad position of each surface is pooled with a
  single learned query (permutation invariant over the set of surfaces).
* Heads: two linear maps from the pooled vector to ``mu`` and ``logvar``; the
  latent ``z`` is reparameterized during training and ``mu`` at inference.
* Decoder: ``ModernBertDecoderModel`` with the latent prepended as a prefix
  position (z-prefix), or the T5 decoder stack with the latent as
  cross-attention memory (``decoder="t5"``). The decoder input is
  ``[BOS, tag, tokens...]`` and the target is ``[tag, tokens...]``; priming
  is the ``tag`` token, so generation seeded with ``[gk]`` yields the greek
  variant of the person encoded in ``z``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Config
from transformers.models.modernbert.configuration_modernbert import ModernBertConfig
from transformers.models.modernbert.modeling_modernbert import ModernBertModel
from transformers.models.modernbert_decoder.configuration_modernbert_decoder import ModernBertDecoderConfig
from transformers.models.modernbert_decoder.modeling_modernbert_decoder import ModernBertDecoderModel
from transformers.models.t5.modeling_t5 import T5Stack

from .config_classes import ConfigForModel


class OdinModel(nn.Module):
    """Encoder-decoder VAE over sets of name surfaces."""

    def __init__(
        self,
        config: ConfigForModel,
        *,
        vocab_size: int,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
    ):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.decoder_family: str = config.decoder
        d = config.hidden_size

        enc_cfg = ModernBertConfig(
            vocab_size=vocab_size,
            hidden_size=d,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.encoder_layers,
            num_attention_heads=config.attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            local_attention=config.local_attention,
            pad_token_id=pad_token_id,
        )
        self.encoder = ModernBertModel(enc_cfg)
        self.pool_query = nn.Parameter(torch.empty(d))
        nn.init.normal_(self.pool_query, std=0.02)
        self.mu_head = nn.Linear(d, d)
        self.logvar_head = nn.Linear(d, d)

        if config.decoder == "modernbert":
            dec_cfg = ModernBertDecoderConfig(
                vocab_size=vocab_size,
                hidden_size=d,
                intermediate_size=config.intermediate_size,
                num_hidden_layers=config.decoder_layers,
                num_attention_heads=config.attention_heads,
                max_position_embeddings=config.max_position_embeddings,
                local_attention=config.local_attention,
                pad_token_id=pad_token_id,
            )
            self.decoder = ModernBertDecoderModel(dec_cfg)
            self.z_proj = nn.Linear(d, d, bias=False)
        elif config.decoder == "t5":
            t5_cfg = T5Config(
                vocab_size=vocab_size,
                d_model=d,
                d_kv=d // config.attention_heads,
                d_ff=config.intermediate_size,
                num_layers=config.decoder_layers,
                num_heads=config.attention_heads,
                relative_attention_num_buckets=32,
                relative_attention_max_distance=config.max_position_embeddings,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            t5_cfg.is_decoder = True
            t5_cfg.use_cache = True
            self.decoder = T5Stack(t5_cfg)
            self.z_proj = None
        else:  # pragma: no cover - guarded by pydantic Literal
            raise ValueError(f"Unknown decoder family: {config.decoder!r}")

        # Weight tying: encoder input == decoder input == LM head (T5-style).
        shared = self._decoder_embed_module().weight
        self.encoder.embeddings.tok_embeddings.weight = shared
        self.lm_head = nn.Linear(d, vocab_size, bias=False)
        self.lm_head.weight = shared

    def _decoder_embed_module(self) -> nn.Embedding:
        """The decoder's token-embedding module (family-dependent)."""
        decoder = self.decoder
        if self.decoder_family == "t5":
            assert isinstance(decoder, T5Stack)
            return decoder.embed_tokens
        assert isinstance(decoder, ModernBertDecoderModel)
        return decoder.embeddings.tok_embeddings

    # ------------------------------------------------------------------ #
    # Encoder side
    # ------------------------------------------------------------------ #
    def _cluster_pool(self, v: torch.Tensor, k_per_cluster: torch.Tensor) -> torch.Tensor:
        """Permutation-invariant PMA pooling of variable-size clusters.

        Args:
            v: ``(N, d)`` per-surface vectors, rows laid out cluster-major.
            k_per_cluster: ``(B,)`` surfaces per cluster (``sum == N``).

        The learned query is scored per row; the softmax is taken per cluster
        (masked to the cluster's own rows, stabilized by its max score), so
        clusters with different surface counts pool correctly.
        """
        n, dim = v.shape
        b = k_per_cluster.shape[0]
        device = v.device
        starts = torch.cat([torch.zeros(1, dtype=torch.long, device=device), k_per_cluster.cumsum(0)[:-1]])
        arange = torch.arange(n, device=device)
        cluster_of_row = torch.searchsorted(starts, arange, right=True) - 1
        scores = (v * self.pool_query).sum(-1) / (dim**0.5)
        cluster_max = torch.full((b,), float("-inf"), device=device).scatter_reduce(
            0, cluster_of_row, scores, reduce="amax", include_self=False
        )
        exp_s = torch.exp(scores - cluster_max[cluster_of_row])
        w_sum = torch.zeros(b, device=device).index_add_(0, cluster_of_row, exp_s)
        weights = exp_s / w_sum[cluster_of_row]
        return torch.zeros(b, dim, device=device).index_add_(0, cluster_of_row, weights.unsqueeze(-1) * v)

    def encode(
        self,
        surf_ids: torch.Tensor,
        surf_mask: torch.Tensor,
        *,
        k_per_cluster: torch.Tensor | None = None,
        k: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of surfaces into ``(mu, logvar)``.

        Args:
            surf_ids: ``(N, L)`` token ids, tag token prepended per surface,
                rows laid out cluster-major.
            surf_mask: ``(N, L)`` with 1 for real tokens (including the tag).
            k_per_cluster: ``(B,)`` surfaces per cluster (``sum == N``).
            k: Uniform shortcut for ``k_per_cluster`` (rows ``b*k..b*k+k-1``
                form cluster ``b``); ``None`` with ``k_per_cluster=None``
                treats all rows as one cluster.

        Returns:
            ``(mu, logvar)`` of shape ``(B, d)``.
        """
        n = surf_ids.shape[0]
        if k_per_cluster is None:
            if k is None:
                k_per_cluster = torch.full((1,), n, device=surf_ids.device, dtype=torch.long)
            else:
                if k <= 0 or n % k != 0:
                    raise ValueError(f"n ({n}) must be a positive multiple of k ({k}).")
                k_per_cluster = torch.full((n // k,), k, device=surf_ids.device, dtype=torch.long)
        else:
            k_per_cluster = k_per_cluster.to(surf_ids.device, dtype=torch.long)
            if int(k_per_cluster.sum()) != n:
                raise ValueError(f"sum(k_per_cluster) ({int(k_per_cluster.sum())}) != n ({n}).")
        hidden = self.encoder(input_ids=surf_ids, attention_mask=surf_mask).last_hidden_state
        last = surf_mask.flip(1).argmax(1)
        v = hidden[torch.arange(n, device=hidden.device), last]
        pooled = self._cluster_pool(v, k_per_cluster)
        return self.mu_head(pooled), self.logvar_head(pooled)

    # ------------------------------------------------------------------ #
    # Training forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        surf_ids: torch.Tensor,
        surf_mask: torch.Tensor,
        k_per_cluster: torch.Tensor,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
        target_tags: torch.Tensor,
        target_primed: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Teacher-forced step.

        Args:
            surf_ids / surf_mask: ``(N, L_in)`` encoder inputs (tag prepended),
                rows laid out cluster-major.
            k_per_cluster: ``(B,)`` surfaces per cluster.
            target_ids / target_mask: ``(NK, L)`` target token ids (no tag).
            target_tags: ``(NK,)`` tag token id of each target's true script.
            target_primed: ``(NK,)`` 0/1 — primed rows are seeded with the tag
                (decoder input ``[BOS, tag, t0, ...]``, labels ``[tag, t0, ...]``);
                unprimed rows decode from the latent alone (``[BOS, t0, ...]``,
                labels ``[t0, ...]``), the unknown-alphabet regime.

        Returns a dict with ``loss`` (CE + ``kl_weight`` * KL), ``ce`` and ``kl``.
        """
        b = k_per_cluster.shape[0]
        mu, logvar = self.encode(surf_ids, surf_mask, k_per_cluster=k_per_cluster)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

        k_out = target_ids.shape[0] // b
        z_rep = z.repeat_interleave(k_out, dim=0)  # (NK, d)
        device = target_ids.device

        nk = target_ids.shape[0]
        primed = target_primed.bool().to(device)
        tags_col = target_tags.unsqueeze(1)
        body = target_ids[:, :-1]
        bos_col = torch.full((nk, 1), self.bos_token_id, dtype=torch.long, device=device)
        pad_col = torch.zeros(nk, 1, dtype=torch.long, device=device)
        # Primed:   [BOS, tag, t0, t1, ...]      Unprimed: [BOS, t0, t1, ..., pad]
        dec_primed = torch.cat([bos_col, tags_col, body], dim=1)
        dec_unprimed = torch.cat([bos_col, body, pad_col], dim=1)
        decoder_input = torch.where(primed.unsqueeze(1), dec_primed, dec_unprimed)
        # Primed:   labels [tag, t0, t1, ...]    Unprimed: labels [t0, t1, ...]
        lab_primed = torch.cat([tags_col, target_ids], dim=1)
        # Unprimed: input [BOS, t0, ...] predicts [t0, t1, ...] — the labels align
        # with target_ids directly (a pad prefix here would shift the whole
        # sequence back by one and train P(PAD | z, BOS) at the first
        # generated position). Trailing pad keeps the shape equal to the
        # primed labels, which carry the extra tag column.
        lab_unprimed = torch.cat([target_ids, pad_col], dim=1)
        labels = torch.where(primed.unsqueeze(1), lab_primed, lab_unprimed)
        real_len = target_mask.sum(1)  # (NK,) real target tokens per row
        label_len = torch.where(primed, real_len + 1, real_len)
        label_mask = torch.arange(labels.shape[1], device=device) < label_len.unsqueeze(1)
        labels = labels.masked_fill(label_mask == 0, -100)
        dec_len = label_len + 1  # + BOS
        dec_mask = torch.arange(decoder_input.shape[1], device=device) < dec_len.unsqueeze(1)

        if self.decoder_family == "modernbert":
            assert self.z_proj is not None
            embeddings = torch.cat(
                [self.z_proj(z_rep).unsqueeze(1), self._decoder_embed_module()(decoder_input)], dim=1
            )
            position_ids = (
                torch.arange(embeddings.shape[1], device=device).unsqueeze(0).expand(decoder_input.shape[0], -1)
            )
            logits = self.lm_head(self.decoder(inputs_embeds=embeddings, position_ids=position_ids).last_hidden_state)[
                :, 1:
            ]
        else:
            memory = z_rep.unsqueeze(1)
            memory_mask = torch.ones(memory.shape[0], 1, dtype=torch.long, device=device)
            logits = self.lm_head(
                self.decoder(
                    input_ids=decoder_input,
                    attention_mask=dec_mask,
                    encoder_hidden_states=memory,
                    encoder_attention_mask=memory_mask,
                ).last_hidden_state
            )

        ce = F.cross_entropy(logits.reshape(-1, self.vocab_size), labels.reshape(-1), ignore_index=-100)
        kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        loss = ce + self.config.kl_weight * kl
        return {"loss": loss, "ce": ce, "kl": kl}

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def log_prob(self, z: torch.Tensor, tag_id: int | None, token_ids: list[int]) -> tuple[float, list[float]]:
        """Deterministic log-probability of one surface under the latent ``z``.

        The inference-time counterpart of ``forward`` without reparameterization
        noise (``z`` is used as-is, i.e. ``mu``) and without a batch: the surface
        is teacher-forced in a single decoder pass.

        Primed (``tag_id`` given): decoder input ``[BOS, tag, t0, ..., t_{n-2}]``,
        predicting ``[tag, t0, ..., t_{n-1}]`` — the known-alphabet regime.
        Unprimed (``tag_id=None``): decoder input ``[BOS, t0, ..., t_{n-2}]``,
        predicting ``[t0, ..., t_{n-1}]`` — the unknown-alphabet regime, where
        the first predicted token is the script itself.

        Args:
            z: Latent vector of shape ``(d,)`` or ``(1, d)`` (use ``mu``).
            tag_id: Priming script tag token id, or ``None`` for unprimed.
            token_ids: Surface token ids (no tag, no EOS).

        Returns:
            ``(total_logprob, per_token_logprobs)`` aligned with the predicted
            tokens (``n+1`` entries when primed, ``n`` when unprimed).
        """
        ids = [int(t) for t in token_ids]
        if tag_id is None:
            prefix = [self.bos_token_id]
            targets = ids
        else:
            prefix = [self.bos_token_id, int(tag_id)]
            targets = [int(tag_id)] + ids
        dec_ids = prefix + ids[:-1]  # teacher-force up to the penultimate token
        if len(targets) == 0:
            raise ValueError("token_ids must contain at least one token.")

        was_training = self.training
        if was_training:
            self.eval()
        try:
            with torch.no_grad():
                z = z.view(1, -1).to(next(self.parameters()).device)
                device = z.device
                if self.decoder_family == "modernbert":
                    assert self.z_proj is not None
                    embed = self._decoder_embed_module()
                    emb = torch.cat(
                        [
                            self.z_proj(z).unsqueeze(1),
                            embed(torch.tensor([dec_ids], dtype=torch.long, device=device)),
                        ],
                        dim=1,
                    )
                    position_ids = torch.arange(emb.shape[1], device=device).unsqueeze(0)
                    hidden = self.decoder(inputs_embeds=emb, position_ids=position_ids).last_hidden_state
                    logits = self.lm_head(hidden)[:, 1:]
                else:
                    # T5 takes the latent as cross-attention memory, so its
                    # hidden states align 1:1 with dec_ids (no z prefix to skip).
                    memory = z.unsqueeze(1)
                    memory_mask = torch.ones(1, 1, dtype=torch.long, device=device)
                    hidden = self.decoder(
                        input_ids=torch.tensor([dec_ids], dtype=torch.long, device=device),
                        encoder_hidden_states=memory,
                        encoder_attention_mask=memory_mask,
                    ).last_hidden_state
                    logits = self.lm_head(hidden)
                log_probs = F.log_softmax(logits, dim=-1)
                per_token = [float(log_probs[0, j, targets[j]]) for j in range(len(targets))]
                return sum(per_token), per_token
        finally:
            if was_training:
                self.train()

    def _sample(
        self, logits: torch.Tensor, temperature: float, top_p: float | None, generator: torch.Generator | None
    ) -> torch.Tensor:
        if temperature <= 0.0:
            return logits.argmax(-1)
        logits = logits / temperature
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = logits.sort(-1, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = probs.cumsum(-1)
            keep = cumulative - probs > top_p
            keep[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(keep, float("-inf"))
            logits = logits.new_full(logits.shape, float("-inf")).scatter_(1, sorted_indices, sorted_logits)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1, generator=generator).squeeze(-1)

    def generate(
        self,
        z: torch.Tensor,
        tag_id: int | None,
        *,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_p: float | None = None,
        generator: torch.Generator | None = None,
    ) -> list[int]:
        """Greedy/sample decoding of one surface.

        Args:
            z: Latent vector of shape ``(d,)`` or ``(1, d)`` (use ``mu``).
            tag_id: Token id of the priming script tag (e.g. ``[gk]``), or
                ``None`` to decode without alphabet information (the
                unknown-alphabet regime).
            max_new_tokens: Hard cap on generated tokens (excluding BOS/tag).
            temperature: 0 for greedy; >0 for sampling.
            top_p: Optional nucleus filter when sampling.
            generator: Optional ``torch.Generator`` for reproducible sampling.

        Returns the generated token ids (no BOS/tag), ending at ``[EOS]``.
        """
        was_training = self.training
        if was_training:
            self.eval()
        try:
            with torch.no_grad():
                z = z.view(1, -1).to(next(self.parameters()).device)
                generated: list[int] = []
                prefix_ids = [self.bos_token_id, int(tag_id)] if tag_id is not None else [self.bos_token_id]
                if self.decoder_family == "modernbert":
                    assert self.z_proj is not None
                    embed = self._decoder_embed_module()
                    prefix = embed(torch.tensor([prefix_ids], device=z.device))
                    embeddings = torch.cat([self.z_proj(z).unsqueeze(1), prefix], dim=1)
                    position_ids = torch.arange(embeddings.shape[1], device=z.device).unsqueeze(0)
                    out = self.decoder(inputs_embeds=embeddings, position_ids=position_ids, use_cache=True)
                    base_pos = embeddings.shape[1]  # position of the first generated token
                    for _ in range(max_new_tokens):
                        token = int(
                            self._sample(self.lm_head(out.last_hidden_state)[:, -1], temperature, top_p, generator)
                        )
                        if token == self.eos_token_id:
                            break
                        generated.append(token)
                        step_ids = torch.tensor([[token]], device=z.device)
                        step_pos = torch.tensor([base_pos + len(generated) - 1], device=z.device).unsqueeze(0)
                        out = self.decoder(
                            inputs_embeds=embed(step_ids),
                            position_ids=step_pos,
                            use_cache=True,
                            past_key_values=out.past_key_values,
                        )
                else:
                    memory = z.unsqueeze(1)
                    memory_mask = torch.ones(1, 1, dtype=torch.long, device=z.device)
                    seed = torch.tensor([prefix_ids], device=z.device)
                    out = self.decoder(
                        input_ids=seed,
                        encoder_hidden_states=memory,
                        encoder_attention_mask=memory_mask,
                        use_cache=True,
                    )
                    for _ in range(max_new_tokens):
                        token = int(
                            self._sample(self.lm_head(out.last_hidden_state)[:, -1], temperature, top_p, generator)
                        )
                        if token == self.eos_token_id:
                            break
                        generated.append(token)
                        out = self.decoder(
                            input_ids=torch.tensor([[token]], device=z.device),
                            encoder_hidden_states=memory,
                            encoder_attention_mask=memory_mask,
                            use_cache=True,
                            past_key_values=out.past_key_values,
                        )
                return generated
        finally:
            if was_training:
                self.train()
