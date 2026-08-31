"""Inference API for the trained Odin name-surface VAE.

Wraps ``OdinModel`` for production-style use: load the checkpoint once, then
encode name clusters into posteriors, score surfaces against them, rank
candidate clusters, and generate surface variants from the posterior.

    from odin_vae.inference import OdinInference

    odin = OdinInference.from_checkpoint("009.ckpt", "runners/odin_vae_config.yaml")
    post = odin.posterior([("la", "J. Smith")])
    nll = odin.score_surface(post.mu, "John Allan Smith", "la")
    ranked = odin.match([("la", "J. Smith")], {"A": [("la", "J. Smith")], ...})
    variants = odin.generate([("la", "J. Smith")], "la", n_samples=16, temperature=0.7)

Surfaces are ``(tag, text)`` pairs with the 12-script tags (``la``, ``gk``,
...). Plain ``str`` lists are accepted when one shared tag is given.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from odin_vae.augment import SCRIPTS
from odin_vae.config_classes import ConfigForRoot
from odin_vae.configid import verify_configid
from odin_vae.model import OdinModel

__all__ = ["MatchResult", "OdinInference", "Posterior"]


def _build_tokenizer(model_cfg):
    """Prefer the frozen C tokenizer; fall back to the HF fast tokenizer.

    Both expose the API used here (encode/decode/convert_tokens_to_ids and the
    pad/bos/eos/unk token ids), so the rest of the class is agnostic.
    """
    if getattr(model_cfg, "use_fast_tokenizer", True):
        try:
            from odin_tokenizer_fast import OdinFastTokenizer

            return OdinFastTokenizer(str(Path(model_cfg.tokenizer_path) / "tokenizer.json"))
        except Exception:  # noqa: BLE001  # extension missing or vocab mismatch -> HF
            pass
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast.from_pretrained(model_cfg.tokenizer_path)  # nosec: B615


def _normalize_surfaces(surfaces: Sequence[tuple[str, str]] | Sequence[str], tag: str | None) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for s in surfaces:
        if isinstance(s, tuple):
            t, text = s
            out.append((t, text))
        else:
            if tag is None:
                raise ValueError("A shared `tag` is required when surfaces are plain strings.")
            out.append((tag, s))
    for t, _ in out:
        if t not in SCRIPTS:
            raise ValueError(f"Unknown script tag {t!r}; expected one of {sorted(SCRIPTS)}.")
    if not out:
        raise ValueError("At least one surface is required.")
    return out


@dataclass(frozen=True)
class Posterior:
    """The VAE posterior ``q(z | cluster) = N(mu, diag(exp(logvar)))``."""

    mu: torch.Tensor
    logvar: torch.Tensor

    @property
    def std(self) -> torch.Tensor:
        return torch.exp(0.5 * self.logvar)

    def kl_to_prior(self) -> float:
        """KL(q || N(0, 1)) in nats per latent dimension."""
        return float(-0.5 * torch.mean(1.0 + self.logvar - self.mu.pow(2) - self.logvar.exp()))

    def mean_variance(self) -> float:
        return float(self.logvar.exp().mean())

    @property
    def d(self) -> int:
        return int(self.mu.shape[0])


@dataclass(frozen=True)
class MatchResult:
    """One candidate cluster's standing in a :meth:`OdinInference.match` ranking."""

    name: str
    score: float
    margin: float

    def __str__(self) -> str:
        return f"{self.name}\tscore={self.score:.4f}\tmargin={self.margin:+.4f}"


def _load_state_dict(checkpoint: str | Path) -> dict[str, Any]:
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    state = ckpt.get("state_dict", ckpt)
    prefix = "model."
    if state and all(str(k).startswith(prefix) for k in state):
        state = {str(k)[len(prefix) :]: v for k, v in state.items()}
    return state


class OdinInference:
    """Load-once inference handle over a trained :class:`OdinModel`."""

    def __init__(
        self,
        model: OdinModel,
        tokenizer: Any,
        *,
        device: str | None = None,
        max_surfaces: int = 12,
        max_tokens: int = 63,
    ):
        if max_surfaces < 1:
            raise ValueError("max_surfaces must be >= 1.")
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1.")
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_surfaces = max_surfaces
        self.max_tokens = max_tokens
        self.model.eval().to(self.device)
        self._tag_ids: dict[str, int] | None = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        config: str | Path | ConfigForRoot,
        *,
        device: str | None = None,
        max_surfaces: int = 12,
        max_tokens: int = 63,
    ) -> OdinInference:
        """Build from a Lightning checkpoint plus the training config yaml."""

        if isinstance(config, ConfigForRoot):
            root_cfg = config
        else:
            import yaml

            with open(config, encoding="utf-8") as handle:
                root_cfg = ConfigForRoot.from_mapping(yaml.safe_load(handle))
        tokenizer = _build_tokenizer(root_cfg.model)
        model = OdinModel(
            root_cfg.model,
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        verify_configid(model, checkpoint)
        model.load_state_dict(_load_state_dict(checkpoint), strict=True)
        return cls(model, tokenizer, device=device, max_surfaces=max_surfaces, max_tokens=max_tokens)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @property
    def tag_ids(self) -> dict[str, int]:
        if self._tag_ids is None:
            self._tag_ids = {tag: self.tokenizer.convert_tokens_to_ids(f"[{tag}]") for tag in SCRIPTS}
        return self._tag_ids

    def _encoder_rows(self, surfaces: list[tuple[str, str]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encoder inputs for one cluster (harness protocol: ``[tag]`` prefix per
        surface, ``max_tokens`` cap, right padding). Returns ``(surf_ids, surf_mask)``."""
        rows = surfaces[: self.max_surfaces]
        ids = [
            [self.tag_ids[tag]] + self.tokenizer.encode(text, add_special_tokens=False)[: self.max_tokens]
            for tag, text in rows
        ]
        length = max(len(row) for row in ids)
        surf_ids = torch.full((len(ids), length), self.model.pad_token_id, dtype=torch.long)
        surf_mask = torch.zeros((len(ids), length), dtype=torch.long)
        for i, row in enumerate(ids):
            surf_ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
            surf_mask[i, : len(row)] = 1
        return surf_ids, surf_mask

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def posterior(self, surfaces: Sequence[tuple[str, str]] | Sequence[str], tag: str | None = None) -> Posterior:
        """Encode a cluster's surfaces into its posterior ``(mu, logvar)``."""
        rows = _normalize_surfaces(surfaces, tag)
        surf_ids, surf_mask = self._encoder_rows(rows)
        k = torch.full((1,), len(rows), dtype=torch.long)
        with torch.no_grad():
            mu, logvar = self.model.encode(
                surf_ids.to(self.device), surf_mask.to(self.device), k_per_cluster=k.to(self.device)
            )
        return Posterior(mu=mu[0].cpu(), logvar=logvar[0].cpu())

    def score_surface(self, mu: torch.Tensor, text: str, tag: str, *, primed: bool = True) -> float:
        """Nats-per-token (NLL/tok, positive) of ``text`` under ``mu``, primed or unprimed."""
        if tag not in SCRIPTS:
            raise ValueError(f"Unknown script tag {tag!r}.")
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            raise ValueError("Surface text tokenizes to no tokens.")
        tag_id = self.tag_ids[tag] if primed else None
        z = mu.to(self.device).float()
        lp, per = self.model.log_prob(z, tag_id, ids)
        return -lp / len(per)

    def score_cluster(
        self,
        mu: torch.Tensor,
        surfaces: Sequence[tuple[str, str]] | Sequence[str],
        tag: str | None = None,
        *,
        primed: bool = True,
    ) -> float:
        """Mean NLL/tok of the cluster's surfaces under ``mu`` (its match score)."""
        rows = _normalize_surfaces(surfaces, tag)
        scores = [self.score_surface(mu, text, t, primed=primed) for t, text in rows]
        return sum(scores) / len(scores)

    def match(
        self,
        query: Sequence[tuple[str, str]] | Sequence[str],
        candidates: Mapping[str, Sequence[tuple[str, str]] | Sequence[str]],
        tag: str | None = None,
        *,
        primed: bool = True,
    ) -> list[MatchResult]:
        """Rank candidate clusters by match score (mean NLL/tok, lower = better).

        ``margin`` is the gap between the candidate's score and the median score
        of all other candidates (positive = distinctly better than the field).
        """
        if not candidates:
            raise ValueError("candidates must be non-empty.")
        mu = self.posterior(query, tag).mu
        scored = [(name, self.score_cluster(mu, cands, tag, primed=primed)) for name, cands in candidates.items()]
        scored.sort(key=lambda item: item[1])
        scores = sorted(item[1] for item in scored)
        results: list[MatchResult] = []
        for rank, (name, score) in enumerate(scored):
            others = scores[:rank] + scores[rank + 1 :]
            median = (
                others[len(others) // 2]
                if len(others) % 2
                else 0.5 * (others[len(others) // 2 - 1] + others[len(others) // 2])
            )
            results.append(MatchResult(name=name, score=score, margin=median - score))
        return results

    def decode(
        self,
        surfaces: Sequence[tuple[str, str]] | Sequence[str],
        tag: str,
        *,
        max_new_tokens: int = 48,
    ) -> str:
        """Deterministic (posterior-mode) decode of one surface in ``tag``."""
        mu = self.posterior(surfaces, tag).mu
        ids = self.model.generate(mu.to(self.device), self.tag_ids[tag], max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def generate(
        self,
        surfaces: Sequence[tuple[str, str]] | Sequence[str],
        tag: str,
        *,
        n_samples: int = 1,
        max_new_tokens: int = 48,
        temperature: float = 0.0,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> list[str]:
        """Sample ``n_samples`` surface variants from the posterior (decoded in ``tag``)."""
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1.")
        rows = _normalize_surfaces(surfaces, tag)
        surf_ids, surf_mask = self._encoder_rows(rows)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        with torch.no_grad():
            samples, _, _ = self.model.sample_generate(
                surf_ids.to(self.device),
                surf_mask.to(self.device),
                k=len(rows),
                tag_id=self.tag_ids[tag],
                n_samples=n_samples,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                generator=generator,
            )
        return [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in samples]

    def ambiguity(
        self, surfaces: Sequence[tuple[str, str]] | Sequence[str], tag: str | None = None
    ) -> dict[str, float]:
        """Summary statistics of the posterior for a cluster (its ambiguity signal)."""
        post = self.posterior(surfaces, tag)
        return {
            "kl_to_prior": post.kl_to_prior(),
            "mean_variance": post.mean_variance(),
            "median_std": float(torch.median(post.std).item()),
            "max_variance": float(post.logvar.exp().max()),
            "dims_over_prior": float((post.logvar.exp() > 1.0).float().mean()),
        }
