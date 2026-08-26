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
    def _pool(self, v: torch.Tensor) -> torch.Tensor:
        """Permutation-invariant pooling of per-surface vectors ``(B, K, d)``."""
        dim = v.shape[-1]
        scores = (v * self.pool_query).sum(-1) / (dim**0.5)
        weights = F.softmax(scores, dim=1)
        return (weights.unsqueeze(-1) * v).sum(1)

    def encode(
        self,
        surf_ids: torch.Tensor,
        surf_mask: torch.Tensor,
        *,
        k: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of surfaces into ``(mu, logvar)``.

        Args:
            surf_ids: ``(N, L)`` token ids, tag token prepended per surface.
            surf_mask: ``(N, L)`` with 1 for real tokens (including the tag).
            k: Surfaces per cluster (rows ``b*k .. b*k+k-1`` form cluster
                ``b``); ``None`` treats all rows as one cluster.

        Returns:
            ``(mu, logvar)`` of shape ``(N // k, d)``.
        """
        n = surf_ids.shape[0]
        if k is None:
            k = n
        if k <= 0 or n % k != 0:
            raise ValueError(f"n ({n}) must be a positive multiple of k ({k}).")
        hidden = self.encoder(input_ids=surf_ids, attention_mask=surf_mask).last_hidden_state
        last = surf_mask.flip(1).argmax(1)
        v = hidden[torch.arange(n, device=hidden.device), last]
        pooled = self._pool(v.view(-1, k, hidden.shape[-1]))
        return self.mu_head(pooled), self.logvar_head(pooled)

    # ------------------------------------------------------------------ #
    # Training forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        surf_ids: torch.Tensor,
        surf_mask: torch.Tensor,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
        target_tags: torch.Tensor,
        n_clusters: int,
    ) -> dict[str, torch.Tensor]:
        """Teacher-forced step.

        Args:
            surf_ids / surf_mask: ``(B*K_in, L_in)`` encoder inputs (tag prepended).
            target_ids / target_mask: ``(B*K_out, L)`` target token ids (no tag).
            target_tags: ``(B*K_out,)`` tag token id priming each target.
            n_clusters: ``B`` (surfaces/targets are laid out cluster-major).

        Returns a dict with ``loss`` (CE + ``kl_weight`` * KL), ``ce`` and ``kl``.
        """
        b = int(n_clusters)
        k_in = surf_ids.shape[0] // b
        hidden = self.encoder(input_ids=surf_ids, attention_mask=surf_mask).last_hidden_state
        last = surf_mask.flip(1).argmax(1)
        v = hidden[torch.arange(surf_ids.shape[0], device=hidden.device), last].view(b, k_in, -1)
        pooled = self._pool(v)
        mu = self.mu_head(pooled)
        logvar = self.logvar_head(pooled)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

        k_out = target_ids.shape[0] // b
        z_rep = z.repeat_interleave(k_out, dim=0)  # (B*K_out, d)
        device = target_ids.device

        nk = target_ids.shape[0]
        tags_col = target_tags.unsqueeze(1)
        labels = torch.cat([tags_col, target_ids], dim=1)  # (NK, L+1)
        labels_mask = torch.cat([torch.ones(nk, 1, dtype=torch.long, device=device), target_mask], dim=1)
        labels = labels.masked_fill(labels_mask == 0, -100)
        bos_col = torch.full((target_ids.shape[0], 1), self.bos_token_id, dtype=torch.long, device=device)
        decoder_input = torch.cat([bos_col, tags_col, target_ids[:, :-1]], dim=1)  # (NK, L+1)

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
            input_mask = torch.cat(
                [torch.ones(decoder_input.shape[0], 2, dtype=torch.long, device=device), target_mask[:, :-1]],
                dim=1,
            )
            logits = self.lm_head(
                self.decoder(
                    input_ids=decoder_input,
                    attention_mask=input_mask,
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
        tag_id: int,
        *,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_p: float | None = None,
        generator: torch.Generator | None = None,
    ) -> list[int]:
        """Greedy/sample decoding of one surface primed by ``tag_id``.

        Args:
            z: Latent vector of shape ``(d,)`` or ``(1, d)`` (use ``mu``).
            tag_id: Token id of the priming script tag (e.g. ``[gk]``).
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
                if self.decoder_family == "modernbert":
                    assert self.z_proj is not None
                    embed = self._decoder_embed_module()
                    prefix = embed(torch.tensor([[self.bos_token_id, int(tag_id)]], device=z.device))
                    embeddings = torch.cat([self.z_proj(z).unsqueeze(1), prefix], dim=1)
                    position_ids = torch.arange(embeddings.shape[1], device=z.device).unsqueeze(0)
                    out = self.decoder(inputs_embeds=embeddings, position_ids=position_ids, use_cache=True)
                    for _ in range(max_new_tokens):
                        token = int(
                            self._sample(self.lm_head(out.last_hidden_state)[:, -1], temperature, top_p, generator)
                        )
                        if token == self.eos_token_id:
                            break
                        generated.append(token)
                        step_ids = torch.tensor([[token]], device=z.device)
                        step_pos = torch.tensor([3 + len(generated) - 1], device=z.device).unsqueeze(0)
                        out = self.decoder(
                            inputs_embeds=embed(step_ids),
                            position_ids=step_pos,
                            use_cache=True,
                            past_key_values=out.past_key_values,
                        )
                else:
                    memory = z.unsqueeze(1)
                    memory_mask = torch.ones(1, 1, dtype=torch.long, device=z.device)
                    seed = torch.tensor([[self.bos_token_id, int(tag_id)]], device=z.device)
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
