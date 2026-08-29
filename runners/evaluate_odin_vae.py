"""Evaluate an Odin VAE checkpoint: per-domain val loss + decoding spot-check.

Two measurements over the val shards:

1. **Val loss** over a key range (e.g. the inventor rows 0..3997 or the
   company rows 3998..5995 of the v2 val set): the same collator and forward
   pass as training validation (deterministic per-sample RNG), aggregated
   token-weighted for CE and cluster-weighted for KL.

2. **Decoding spot-check** on held-out clusters from the same range: encode
   the first ``--k-input`` surfaces, then for every remaining surface report
   the teacher-forced log-probability unprimed (unknown-alphabet regime, the
   filter's use case) and primed with the true script tag, plus a greedy
   primed decode and whether it matches the true surface.

Usage:
    uv run python runners/evaluate_odin_vae.py \
        --config runners/odin_vae_config.yaml \
        --checkpoint lightning_logs/odin_vae/version_18/002.ckpt \
        [--key-range 0:3998] [--n-spot 40] [--k-input 2] [--seed 123] \
        [--label v18-inventor] [--out report.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import torch
import yaml

RUNNER_DIR = Path(__file__).resolve().parent
if str(RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(RUNNER_DIR))

import helpers  # noqa: E402
from odin_vae.augment import SCRIPTS, LetterAugmenter  # noqa: E402
from odin_vae.config_classes import ConfigForRoot  # noqa: E402
from odin_vae.data.adapters import ClusterSampleAdapter  # noqa: E402
from odin_vae.data.collators import OdinVAECollator  # noqa: E402
from odin_vae.model import OdinModel  # noqa: E402
from run_train_odin_vae import _setup_tokenizer  # noqa: E402

log = logging.getLogger(__name__)

_TAG_PREFIX = "["


def key_index(key: str) -> int:
    """``val-000000123`` -> 123 (shard member names may carry a ``.json`` suffix)."""
    base = key.rsplit(".", 1)[0]
    return int(base.rsplit("-", 1)[-1])


def in_range(key: str, lo: int, hi: int) -> bool:
    return lo <= key_index(key) < hi


def load_model(root_cfg: ConfigForRoot, tokenizer, checkpoint: Path) -> OdinModel:
    model = OdinModel(
        root_cfg.model,
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    state = ckpt.get("state_dict", ckpt)
    prefix = "model."
    if state and all(str(k).startswith(prefix) for k in state):
        state = {str(k)[len(prefix) :]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model


def cluster_inputs(
    model: OdinModel, tokenizer, tags: list[str], cells: list[str]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build encoder inputs for one cluster (harness protocol: tag prefix, 12-surface
    cap, 63-token cap, right padding). Returns ``(surf_ids, surf_mask, k_per_cluster)``."""
    rows = list(zip(tags, cells))[:12]
    ids = [tokenizer.encode(text, add_special_tokens=False)[:63] for _, text in rows]
    ids = [[tokenizer.convert_tokens_to_ids(f"{_TAG_PREFIX}{tag}]")] + row for tag, row in zip(tags, ids)]
    length = max(len(row) for row in ids)
    surf_ids = torch.full((len(ids), length), model.pad_token_id, dtype=torch.long)
    surf_mask = torch.zeros((len(ids), length), dtype=torch.long)
    for i, row in enumerate(ids):
        surf_ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
        surf_mask[i, : len(row)] = 1
    k_per_cluster = torch.full((1,), len(ids), dtype=torch.long)
    return surf_ids, surf_mask, k_per_cluster


def encode_cluster(model: OdinModel, tokenizer, tags: list[str], cells: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a cluster's surfaces (clean) into its ``(mu, logvar)`` (harness protocol)."""
    surf_ids, surf_mask, k_per_cluster = cluster_inputs(model, tokenizer, tags, cells)
    device = next(model.parameters()).device
    with torch.no_grad():
        mu, logvar = model.encode(surf_ids.to(device), surf_mask.to(device), k_per_cluster=k_per_cluster)
    return mu[0], logvar[0]


def encode_inputs(model: OdinModel, tokenizer, tags: list[str], cells: list[str]) -> torch.Tensor:
    """Encode a cluster's surfaces (clean) into its mu vector (harness protocol)."""
    mu, _ = encode_cluster(model, tokenizer, tags, cells)
    return mu


def run_val_loss(
    model: OdinModel,
    collator: OdinVAECollator,
    records: list[dict],
    batch_size: int,
    kl_weight: float,
    device: str,
) -> dict:
    """Collate + forward over the given records; token-weighted CE, cluster-weighted KL."""
    sum_ce_tok = 0.0
    n_tok = 0
    sum_kl_cl = 0.0
    n_cl = 0
    batch: list[dict] = []

    def flush() -> None:
        nonlocal sum_ce_tok, n_tok, sum_kl_cl, n_cl, batch
        if not batch:
            return
        tensors = collator(batch)
        batch = []
        b = len(tensors["k_per_cluster"])
        n_cl += b
        with torch.no_grad():
            outputs = model(**{k: v.to(device) if torch.is_tensor(v) else v for k, v in tensors.items()})
        n_tokens = int(tensors["target_mask"].sum())
        sum_ce_tok += float(outputs["ce"]) * n_tokens
        n_tok += n_tokens
        sum_kl_cl += float(outputs["kl"]) * b

    for rec in records:
        batch.append(rec)
        if len(batch) >= batch_size:
            flush()
    flush()
    mean_ce = sum_ce_tok / n_tok if n_tok else float("nan")
    mean_kl = sum_kl_cl / n_cl if n_cl else float("nan")
    return {
        "n_clusters": n_cl,
        "n_target_tokens": n_tok,
        "val_ce": mean_ce,
        "val_kl": mean_kl,
        "val_loss": mean_ce + kl_weight * mean_kl,
        "n_truncated_cells": collator.n_truncated,
    }


def run_spot_check(
    model: OdinModel,
    tokenizer,
    records: list[dict],
    *,
    n_spot: int,
    k_input: int,
    seed: int,
    device: str,
    max_new_tokens: int = 48,
) -> dict:
    """Log-prob + greedy-decode check of held-out surfaces in sampled clusters."""
    candidates = [r for r in records if len(r["cells"]) > 1]
    rng = random.Random(seed)
    picks = rng.sample(range(len(candidates)), min(n_spot, len(candidates)))
    tag_ids = {tag: tokenizer.convert_tokens_to_ids(f"{_TAG_PREFIX}{tag}]") for tag in SCRIPTS}
    model = model.to(device)
    summary: dict[str, dict[str, dict]] = {}
    examples: list[dict] = []
    for p in picks:
        rec = candidates[p]
        inputs = list(zip(rec["tags"], rec["cells"]))[:k_input]
        held = list(zip(rec["tags"], rec["cells"]))[k_input:]
        if not held:
            continue
        mu = encode_inputs(model, tokenizer, [t for t, _ in inputs], [c for _, c in inputs])
        row: dict = {"key": rec["key"], "inputs": [{"tag": t, "cell": c} for t, c in inputs], "held": []}
        for tag, cell in held:
            ids = tokenizer.encode(cell, add_special_tokens=False)
            lp_un, per_un = model.log_prob(mu, None, ids)
            lp_pr, per_pr = model.log_prob(mu, tag_ids[tag], ids)
            gen_ids = model.generate(mu, tag_ids[tag], max_new_tokens=max_new_tokens, temperature=0.0)
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            row["held"].append(
                {
                    "tag": tag,
                    "cell": cell,
                    "logp_unprimed": lp_un,
                    "logp_unprimed_per_token": lp_un / len(per_un),
                    "logp_primed": lp_pr,
                    "logp_primed_per_token": lp_pr / len(per_pr),
                    "greedy": gen_text,
                    "match": gen_text == cell,
                }
            )
            agg = summary.setdefault(tag, {"n": 0, "lp_un": 0.0, "lp_pr": 0.0, "matches": 0})
            agg["n"] += 1
            agg["lp_un"] += lp_un / len(per_un)
            agg["lp_pr"] += lp_pr / len(per_pr)
            agg["matches"] += int(gen_text == cell)
        examples.append(row)
    table = {
        tag: {
            "n": agg["n"],
            "mean_logp_per_token_unprimed": agg["lp_un"] / agg["n"],
            "mean_logp_per_token_primed": agg["lp_pr"] / agg["n"],
            "greedy_match_rate": agg["matches"] / agg["n"],
        }
        for tag, agg in sorted(summary.items())
    }
    return {"n_clusters": len(examples), "per_script": table, "examples": examples}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate an Odin VAE checkpoint.")
    parser.add_argument("--config", required=True, help="Training config yaml (model + data paths).")
    parser.add_argument("--checkpoint", required=True, help="Lightning checkpoint (.ckpt).")
    parser.add_argument("--key-range", default="0:1000000", help="Val key index range [lo, hi), e.g. 0:3998.")
    parser.add_argument("--n-spot", type=int, default=40, help="Clusters sampled for the decoding spot-check.")
    parser.add_argument("--k-input", type=int, default=2, help="Input surfaces per spot-check cluster.")
    parser.add_argument("--seed", type=int, default=123, help="Spot-check sampling seed.")
    parser.add_argument("--label", default=None, help="Free-form label recorded in the report.")
    parser.add_argument("--out", default=None, help="Optional JSON report path.")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU (slow).")
    args = parser.parse_args(argv)

    helpers.bootstrap_logging(logging.INFO)
    with open(args.config, encoding="utf-8") as handle:
        root_cfg = ConfigForRoot.from_mapping(yaml.safe_load(handle))
    lo, hi = (int(x) for x in args.key_range.split(":"))

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = _setup_tokenizer(root_cfg.model.tokenizer_path)
    model = load_model(root_cfg, tokenizer, Path(args.checkpoint))
    model.eval().to(device)
    log.info("Loaded %s on %s", args.checkpoint, device)

    aug_cfg = root_cfg.augmentation
    freq_path = Path(root_cfg.data_root) / "char_frequencies.json"
    augmenter = LetterAugmenter(
        aug_cfg.rate, json.loads(freq_path.read_text(encoding="utf-8")), aug_cfg.confusion_weight
    )
    collator = OdinVAECollator(
        tokenizer,
        augmenter,
        k_input=aug_cfg.k_input,
        k_latin_target=aug_cfg.k_latin_target,
        k_non_latin_target=aug_cfg.k_non_latin_target,
        max_tokens=aug_cfg.max_surface_tokens,
        seed=root_cfg.seed,
    )

    val_cfg = root_cfg.splits["val"]
    pattern = f"{root_cfg.data_root}/{val_cfg.dataset.pattern}"
    records = [
        rec
        for rec in ClusterSampleAdapter(urls=pattern, seed=root_cfg.seed, is_endless=False)
        if in_range(rec["key"], lo, hi)
    ]
    log.info("val records in key range [%d, %d): %d", lo, hi, len(records))

    val_stats = run_val_loss(model, collator, records, val_cfg.dataloader.batch_size, root_cfg.model.kl_weight, device)
    spot = run_spot_check(
        model, tokenizer, records, n_spot=args.n_spot, k_input=args.k_input, seed=args.seed, device=device
    )

    report = {
        "checkpoint": str(args.checkpoint),
        "label": args.label,
        "key_range": [lo, hi],
        "val": val_stats,
        "spot_check": {k: spot[k] for k in ("n_clusters", "per_script")},
    }
    print(f"=== val (keys {lo}..{hi - 1}, {val_stats['n_clusters']} clusters) ===")
    print(f"  val_loss {val_stats['val_loss']:.6f}   ce {val_stats['val_ce']:.6f}   kl {val_stats['val_kl']:.6e}")
    print("=== spot check (held-out surfaces) ===")
    print(f"  {'tag':<4} {'n':>5} {'logp/tok unprimed':>18} {'logp/tok primed':>17} {'greedy match':>13}")
    for tag, row in spot["per_script"].items():
        print(
            f"  {tag:<4} {row['n']:>5} {row['mean_logp_per_token_unprimed']:>18.3f} "
            f"{row['mean_logp_per_token_primed']:>17.3f} {row['greedy_match_rate']:>13.1%}"
        )
    for ex in spot["examples"][:3]:
        print(f"  example {ex['key']}: inputs={[i['cell'] for i in ex['inputs']]}")
        for h in ex["held"][:6]:
            print(
                f"    [{h['tag']}] {h['cell'][:48]!r}  unprimed {h['logp_unprimed_per_token']:.2f}/tok  "
                f"primed {h['logp_primed_per_token']:.2f}/tok  greedy {'OK ' if h['match'] else 'diff'} {h['greedy'][:48]!r}"
            )
    if args.out:
        full = dict(report, examples=spot["examples"])
        Path(args.out).write_text(json.dumps(full, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
