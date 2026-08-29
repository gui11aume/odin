"""Demo: sample surface variants from a cluster's posterior q(z | surfaces).

Shows how the VAE's uncertainty scales with the information content of the
input cluster: encode a cluster into (mu, logvar), report the posterior's
KL divergence from the N(0,1) prior and its per-dimension variance, then
draw ``n_samples`` reparameterized latents and greedily decode each. A
low-information cluster (e.g. "J. Smith") should carry a wider posterior and
yield more distinct decodes than a high-information one ("John Allan Smith").

A cluster is one --cluster flag; multiple surfaces of the same cluster are
separated by " || ".

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run python runners/demo_sampling.py \
        --config runners/odin_vae_config.yaml \
        --checkpoint lightning_logs/odin_vae/version_22/009.ckpt \
        --cluster "J. Smith" \
        --cluster "John Allan Smith" \
        --cluster "John Allan Smith || J. Allan Smith || J.A. Smith" \
        --n-samples 32 [--unprimed] [--seed 123] [--out report.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter
from pathlib import Path

import torch
import yaml

RUNNER_DIR = Path(__file__).resolve().parent
if str(RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(RUNNER_DIR))

import helpers  # noqa: E402
from evaluate_odin_vae import cluster_inputs, load_model  # noqa: E402
from odin_vae.config_classes import ConfigForRoot  # noqa: E402
from run_train_odin_vae import _setup_tokenizer  # noqa: E402

log = logging.getLogger(__name__)

SEP = " || "


def posterior_stats(mu: torch.Tensor, logvar: torch.Tensor) -> dict:
    """Per-cluster posterior diagnostics against the N(0, 1) prior."""
    var = torch.exp(logvar)
    kl = 0.5 * torch.sum(var + mu.pow(2) - 1.0 - logvar)
    return {
        "kl": float(kl),
        "mean_var": float(var.mean()),
        "p50_var": float(var.median()),
        "p90_var": float(torch.quantile(var, 0.9)),
        "max_var": float(var.max()),
        "total_var": float(var.sum()),
        "dim": int(var.numel()),
    }


def sample_cluster(
    model,
    tokenizer,
    tag: str,
    cells: list[str],
    *,
    tag_id: int | None,
    n_samples: int,
    max_new_tokens: int,
    generator: torch.Generator,
    device: str,
    temperature: float = 0.0,
    top_p: float | None = None,
) -> dict:
    surf_ids, surf_mask, k_per_cluster = cluster_inputs(model, tokenizer, [tag] * len(cells), cells)
    samples, mu, logvar = model.sample_generate(
        surf_ids.to(device),
        surf_mask.to(device),
        k_per_cluster=k_per_cluster.to(device),
        tag_id=tag_id,
        n_samples=n_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        generator=generator,
    )
    texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in samples]
    counts = Counter(texts)
    n = len(texts)
    entropy = -sum(c / n * math.log(c / n) for c in counts.values())
    return {
        "cells": cells,
        "tag": tag,
        "n_samples": n,
        "posterior": posterior_stats(mu, logvar),
        "n_unique": len(counts),
        "normalized_entropy": entropy / math.log(n) if len(counts) > 1 else 0.0,
        "decodes": [{"text": t, "count": c} for t, c in counts.most_common()],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sample surface variants from a cluster posterior.")
    parser.add_argument("--config", required=True, help="Training config yaml (model + data paths).")
    parser.add_argument("--checkpoint", required=True, help="Lightning checkpoint (.ckpt).")
    parser.add_argument(
        "--cluster",
        action="append",
        required=True,
        help="One cluster; surfaces of the same cluster separated by ' || '. Repeat for more clusters.",
    )
    parser.add_argument("--tag", default="la", help="Script tag for every surface (default: la).")
    parser.add_argument("--n-samples", type=int, default=32, help="Latent samples (decodes) per cluster.")
    parser.add_argument("--max-new-tokens", type=int, default=48, help="Decode length cap.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Token sampling temperature (0 = greedy).")
    parser.add_argument(
        "--top-p", type=float, default=None, help="Nucleus sampling probability (with --temperature > 0)."
    )
    parser.add_argument(
        "--unprimed", action="store_true", help="Decode without a script tag (unknown-alphabet regime)."
    )
    parser.add_argument("--seed", type=int, default=123, help="Sampling seed (reproducible).")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU (slow).")
    parser.add_argument("--out", default=None, help="Optional JSON report path.")
    args = parser.parse_args(argv)

    helpers.bootstrap_logging(logging.INFO)
    with open(args.config, encoding="utf-8") as handle:
        root_cfg = ConfigForRoot.from_mapping(yaml.safe_load(handle))
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = _setup_tokenizer(root_cfg.model.tokenizer_path)
    model = load_model(root_cfg, tokenizer, Path(args.checkpoint))
    model.eval().to(device)
    log.info("Loaded %s on %s", args.checkpoint, device)

    tag_id = None if args.unprimed else tokenizer.convert_tokens_to_ids(f"[{args.tag}]")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    mode = "unprimed" if args.unprimed else f"primed [{args.tag}]"

    results = []
    for spec in args.cluster:
        cells = [c.strip() for c in spec.split(SEP) if c.strip()]
        if not cells:
            raise SystemExit(f"empty cluster: {spec!r}")
        row = sample_cluster(
            model,
            tokenizer,
            args.tag,
            cells,
            tag_id=tag_id,
            n_samples=args.n_samples,
            max_new_tokens=args.max_new_tokens,
            generator=generator,
            device=device,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        results.append(row)
        p = row["posterior"]
        label = SEP.join(cells)
        print(f"=== {label!r}  ({len(cells)} surface(s), {mode}, {row['n_samples']} samples) ===")
        print(
            f"  posterior: KL(q||N(0,1)) {p['kl']:.2f} nats | var mean {p['mean_var']:.3f} "
            f"p50 {p['p50_var']:.3f} p90 {p['p90_var']:.3f} max {p['max_var']:.3f} | "
            f"total var {p['total_var']:.1f} / {p['dim']} dims"
        )
        print(
            f"  diversity: {row['n_unique']}/{row['n_samples']} unique  (normalized entropy {row['normalized_entropy']:.2f})"
        )
        for d in row["decodes"][:15]:
            print(f"    {d['count']:>3}x {d['text']!r}")
        if len(row["decodes"]) > 15:
            print(f"    ... {len(row['decodes']) - 15} more distinct decodes")
        print()

    print("=== summary ===")
    print(f"  {'cluster':<48} {'kl':>7} {'mean_var':>9} {'max_var':>8} {'unique':>9} {'nH':>5}")
    for row in results:
        p = row["posterior"]
        label = (SEP.join(row["cells"])[:44] + "...") if len(SEP.join(row["cells"])) > 47 else SEP.join(row["cells"])
        print(
            f"  {label:<48} {p['kl']:>7.2f} {p['mean_var']:>9.3f} {p['max_var']:>8.3f} "
            f"{row['n_unique']}/{row['n_samples']:>4} {row['normalized_entropy']:>5.2f}"
        )

    if args.out:
        report = {
            "checkpoint": str(args.checkpoint),
            "mode": mode,
            "seed": args.seed,
            "n_samples": args.n_samples,
            "clusters": results,
        }
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
