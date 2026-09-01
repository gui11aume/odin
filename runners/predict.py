"""Predict harness: batch inference over a trained Odin VAE checkpoint.

Four modes over a JSONL input file of query clusters, sharing one scaffold
(configid-verified load, input parsing, batched encoding, JSONL output with
a sibling ``.meta.json`` recording the model identity and settings):

* ``embed``    -- encoder only: one posterior (mu, logvar + ambiguity stats)
                  per cluster, each cluster encoded on its own (the same
                  per-cluster regime the evaluate runner uses; the encoder's
                  pooled position depends on the padding width, so clusters
                  must not be batched to keep embeddings deterministic).
* ``variants`` -- encoder + decoder: greedy (posterior-mode) decode plus
                  ``--n-samples`` posterior samples per ``--tags`` script.
* ``score``    -- encoder + decoder (teacher-forced): NLL/tok of every
                  ``--panel`` cluster's surfaces under each query's latent.
* ``match``    -- score every panel member, rank, keep ``--top`` with margins.

Input file (one cluster per line; ``id`` required, or ``name`` for the panel):

    {"id": "q-0001", "surfaces": [{"tag": "la", "text": "J. Smith"},
                                  {"tag": "gk", "text": "Ιωάννης Σ."}]}
    {"id": "q-0002", "surfaces": ["Acme"], "tag": "la"}

Surfaces are ``{"tag", "text"}`` objects, or plain strings plus a shared
``"tag"``. Clusters are capped at the harness protocol (12 surfaces, 63
tokens each); skips/truncations are counted in the meta file.

The scoring regime: ``--primed primed|unprimed|both`` (score) or
``primed|unprimed`` (match). Primed is the known-alphabet regime; unprimed
is the filter's unknown-alphabet regime.

Usage:
    python runners/predict.py --config runners/odin_vae_config.yaml \
        --checkpoint lightning_logs/odin_vae/version_23/009.ckpt \
        --mode embed --input queries.jsonl --out embed.jsonl
    python runners/predict.py --config ... --checkpoint ... \
        --mode match --input queries.jsonl --panel panel.jsonl --top 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

RUNNER_DIR = Path(__file__).resolve().parent
if str(RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(RUNNER_DIR))

import helpers  # noqa: E402
from odin_vae.augment import SCRIPTS  # noqa: E402
from odin_vae.config_classes import ConfigForRoot  # noqa: E402
from odin_vae.inference import OdinInference, Posterior  # noqa: E402

log = logging.getLogger(__name__)

PRIMING_CHOICES = ("primed", "unprimed", "both")
MATCH_PRIMING_CHOICES = ("primed", "unprimed")


@dataclass(frozen=True)
class Cluster:
    """One input cluster: a key plus its (tag, text) surfaces."""

    key: str
    rows: tuple[tuple[str, str], ...]


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
def _parse_surfaces(raw: Any, shared_tag: Any, line_no: int) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for item in raw:
        if isinstance(item, dict):
            tag = str(item.get("tag", "")).strip()
            text = str(item.get("text", "")).strip()
            if not tag or not text:
                raise ValueError(f"line {line_no}: each surface needs a non-empty 'tag' and 'text'.")
            rows.append((tag, text))
        else:
            text = str(item).strip()
            if not text:
                raise ValueError(f"line {line_no}: empty surface string.")
            if not shared_tag:
                raise ValueError(f"line {line_no}: plain-string surfaces require a shared 'tag'.")
            rows.append((str(shared_tag).strip(), text))
    bad = sorted({t for t, _ in rows if t not in SCRIPTS})
    if bad:
        raise ValueError(f"line {line_no}: unknown script tag(s) {bad}; expected one of {list(SCRIPTS)}.")
    return rows


def load_clusters(path: str | Path, key_field: str = "id") -> tuple[list[Cluster], int]:
    """Load a JSONL file of clusters.

    Returns ``(clusters, n_skipped)`` where ``n_skipped`` counts lines with
    an empty ``surfaces`` list (lines without one are an error).
    """
    clusters: list[Cluster] = []
    skipped = 0
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_no}: invalid JSON ({exc}).") from exc
            key = str(rec.get(key_field) or "").strip()
            if not key:
                key = f"line-{line_no}"
            raw = rec.get("surfaces")
            if raw is None:
                raise ValueError(f"line {line_no}: missing 'surfaces'.")
            if not raw:
                skipped += 1
                continue
            rows = _parse_surfaces(raw, rec.get("tag"), line_no)
            clusters.append(Cluster(key=key, rows=tuple(rows)))
    return clusters, skipped


# --------------------------------------------------------------------------- #
# Mode runners (each returns the list of JSONL rows for its mode)
# --------------------------------------------------------------------------- #
def _ambiguity_fields(post: Posterior) -> dict[str, float]:
    return {
        "kl_to_prior": post.kl_to_prior(),
        "mean_variance": post.mean_variance(),
        "median_std": float(torch.median(post.std).item()),
        "max_variance": float(post.logvar.exp().max()),
        "dims_over_prior": float((post.logvar.exp() > 1.0).float().mean()),
    }


def embed_clusters(odin: OdinInference, clusters: list[Cluster]) -> list[dict[str, Any]]:
    """Posterior per cluster, each cluster encoded on its own (see module docstring)."""
    rows: list[dict[str, Any]] = []
    for cluster in clusters:
        post = odin.posterior(list(cluster.rows))
        rows.append(
            {"id": cluster.key, "mu": post.mu.tolist(), "logvar": post.logvar.tolist(), **_ambiguity_fields(post)}
        )
    return rows


def generate_variants(
    odin: OdinInference,
    clusters: list[Cluster],
    *,
    tags: list[str],
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float | None,
    seed: int | None,
) -> list[dict[str, Any]]:
    """Greedy decode plus posterior samples per (cluster, tag)."""
    rows: list[dict[str, Any]] = []
    for cluster in clusters:
        for tag in tags:
            greedy = odin.decode(list(cluster.rows), tag, max_new_tokens=max_new_tokens)
            variants = odin.generate(
                list(cluster.rows),
                tag,
                n_samples=n_samples,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
            )
            rows.append({"id": cluster.key, "tag": tag, "greedy": greedy, "variants": variants})
    return rows


def score_panel(
    odin: OdinInference,
    queries: list[Cluster],
    panel: list[Cluster],
    *,
    priming: str,
) -> list[dict[str, Any]]:
    """NLL/tok of every panel cluster under each query's latent (cross product)."""
    flags = [True, False] if priming == "both" else [priming == "primed"]
    rows: list[dict[str, Any]] = []
    for query in queries:
        mu = odin.posterior(list(query.rows)).mu
        for member in panel:
            for primed in flags:
                nll = odin.score_cluster(mu, list(member.rows), primed=primed)
                rows.append({"id": query.key, "candidate": member.key, "primed": bool(primed), "nll_tok": nll})
    return rows


def match_panel(
    odin: OdinInference,
    queries: list[Cluster],
    panel: list[Cluster],
    *,
    priming: str,
    top: int,
) -> list[dict[str, Any]]:
    """Rank the panel per query (lower NLL/tok = better), keep the top ``top``."""
    candidates = {member.key: list(member.rows) for member in panel}
    rows: list[dict[str, Any]] = []
    for query in queries:
        results = odin.match(list(query.rows), candidates, primed=(priming == "primed"))
        ranked = [{"name": r.name, "score": r.score, "margin": r.margin} for r in results[:top]]
        rows.append({"id": query.key, "top_name": ranked[0]["name"] if ranked else None, "ranked": ranked})
    return rows


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _count_caps(odin: OdinInference, clusters: list[Cluster]) -> tuple[int, int]:
    """Count surfaces that exceed the token cap and clusters over the surface cap."""
    truncated = sum(
        1
        for c in clusters
        for _, text in c.rows
        if len(odin.tokenizer.encode(text, add_special_tokens=False)) > odin.max_tokens
    )
    dropped = sum(1 for c in clusters if len(c.rows) > odin.max_surfaces)
    return truncated, dropped


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Odin VAE predict harness.")
    parser.add_argument("--config", required=True, help="Training config yaml (model + tokenizer).")
    parser.add_argument("--checkpoint", required=True, help="Lightning checkpoint (.ckpt).")
    parser.add_argument("--mode", required=True, choices=("embed", "variants", "score", "match"))
    parser.add_argument("--input", required=True, help="JSONL file of query clusters.")
    parser.add_argument("--panel", default=None, help="JSONL file of named candidate clusters (score/match).")
    parser.add_argument("--tags", default=None, help="Comma-separated script tags for variants mode (default: all 12).")
    parser.add_argument(
        "--n-samples", type=int, default=1, help="Posterior samples per (cluster, tag) in variants mode."
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for variants mode.")
    parser.add_argument("--top-p", type=float, default=None, help="Optional nucleus filter for variants mode.")
    parser.add_argument("--seed", type=int, default=None, help="Sampling seed for variants mode (reproducibility).")
    parser.add_argument(
        "--primed",
        default="primed",
        choices=PRIMING_CHOICES,
        help="Scoring regime for score/match (default: primed; unprimed is the filter's regime).",
    )
    parser.add_argument("--top", type=int, default=10, help="Candidates kept per query in match mode.")
    parser.add_argument("--device", default=None, help="cuda/cpu (default: cuda if available).")
    parser.add_argument("--out", default=None, help="Output JSONL path (default: <input stem>.<mode>.jsonl).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    helpers.bootstrap_logging(logging.INFO)

    if args.mode in ("score", "match") and not args.panel:
        raise SystemExit(f"--panel is required for {args.mode} mode.")
    if args.mode == "match" and args.primed == "both":
        raise SystemExit("--primed both is not supported for match mode (pick primed or unprimed).")
    if args.mode == "variants" and args.tags is not None:
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
        bad = sorted({t for t in tags if t not in SCRIPTS})
        if bad or not tags:
            raise SystemExit(f"unknown script tag(s) in --tags: {bad or 'empty list'}.")
    else:
        tags = list(SCRIPTS)

    t0 = time.time()
    with open(args.config, encoding="utf-8") as handle:
        root_cfg = ConfigForRoot.from_mapping(yaml.safe_load(handle))
    odin = OdinInference.from_checkpoint(args.checkpoint, root_cfg, device=args.device)
    log.info("Loaded %s (configid=%s) on %s", args.checkpoint, odin.model.configid, odin.device)

    queries, n_skipped = load_clusters(args.input, key_field="id")
    if not queries:
        raise SystemExit(f"no query clusters found in {args.input}.")
    n_truncated, n_dropped = _count_caps(odin, queries)
    panel: list[Cluster] = []
    if args.panel:
        panel, _ = load_clusters(args.panel, key_field="name")
        if not panel:
            raise SystemExit(f"no panel clusters found in {args.panel}.")

    if args.mode == "embed":
        rows = embed_clusters(odin, queries)
    elif args.mode == "variants":
        rows = generate_variants(
            odin,
            queries,
            tags=tags,
            n_samples=args.n_samples,
            max_new_tokens=48,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
        )
    elif args.mode == "score":
        rows = score_panel(odin, queries, panel, priming=args.primed)
    else:
        rows = match_panel(odin, queries, panel, priming=args.primed, top=args.top)

    out = Path(args.out) if args.out else Path(args.input).with_name(f"{Path(args.input).stem}.{args.mode}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta = {
        "configid": odin.model.configid,
        "checkpoint": str(args.checkpoint),
        "mode": args.mode,
        "device": odin.device,
        "n_queries": len(queries),
        "n_panel": len(panel),
        "n_output_rows": len(rows),
        "n_skipped_empty": n_skipped,
        "n_truncated_surfaces": n_truncated,
        "n_clusters_over_surface_cap": n_dropped,
        "primed": args.primed if args.mode in ("score", "match") else None,
        "tags": tags if args.mode == "variants" else None,
        "n_samples": args.n_samples if args.mode == "variants" else None,
        "temperature": args.temperature if args.mode == "variants" else None,
        "top_p": args.top_p if args.mode == "variants" else None,
        "seed": args.seed if args.mode == "variants" else None,
        "seconds": round(time.time() - t0, 2),
    }
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log.info(
        "%s: %d rows -> %s (configid=%s, %.1fs)",
        args.mode,
        len(rows),
        out,
        meta["configid"],
        meta["seconds"],
    )


if __name__ == "__main__":
    main()
