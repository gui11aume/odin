"""Selectivity probe: does the model rank a company's own surfaces above others'?

For each sampled company cluster in a key range (v3 test companies: keys
2000..2999 of the test split), encode the first 2 surfaces into mu, then
score (a) the cluster's remaining held-out cells ("in-cluster") and (b)
cells sampled from other panel companies ("negatives"), both primed with
the cell's true script tag.

    margin = mean(negative NLL/tok) - mean(in-cluster NLL/tok)

A healthy model shows a large positive margin: its own surfaces are far
cheaper to explain than other companies'. This is the filter's core metric.

Usage:
    uv run python runners/selectivity_probe.py \
        --config runners/odin_vae_config.yaml \
        --checkpoint lightning_logs/odin_vae/version_22/009.ckpt \
        [--key-range 2000:3000] [--n-companies 100] [--n-neg-cells 20] [--seed 123]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import yaml

RUNNER_DIR = Path(__file__).resolve().parent
if str(RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(RUNNER_DIR))

import helpers  # noqa: E402
from evaluate_odin_vae import encode_inputs, in_range, load_model  # noqa: E402
from odin_vae.augment import SCRIPTS  # noqa: E402
from odin_vae.config_classes import ConfigForRoot  # noqa: E402
from odin_vae.data.adapters import ClusterSampleAdapter  # noqa: E402
from run_train_odin_vae import _setup_tokenizer  # noqa: E402

log = logging.getLogger(__name__)


def run_probe(model, tokenizer, records: list[dict], *, n_companies: int, n_neg_cells: int, seed: int) -> dict:
    rng = random.Random(seed)
    picks = rng.sample(range(len(records)), min(n_companies, len(records)))
    tag_ids = {tag: tokenizer.convert_tokens_to_ids(f"[{tag}]") for tag in SCRIPTS}

    panel = []
    for p in picks:
        rec = records[p]
        pairs = list(zip(rec["tags"], rec["cells"]))
        if len(pairs) < 4:  # 2 inputs + at least 2 held-out cells
            continue
        inputs, held = pairs[:2], pairs[2:]
        panel.append({"key": rec["key"], "inputs": inputs, "held": held})

    # Negative pool: all cells of the other panel companies.
    all_cells = {row["key"]: [(t, c) for t, c in row["held"]] for row in panel}

    rows = []
    for row in panel:
        mu = encode_inputs(model, tokenizer, [t for t, _ in row["inputs"]], [c for _, c in row["inputs"]])
        in_nll: list[float] = []
        for tag, cell in row["held"]:
            ids = tokenizer.encode(cell, add_special_tokens=False)
            lp, per = model.log_prob(mu, tag_ids[tag], ids)
            in_nll.append(-lp / len(per))
        others = [c for k, cs in all_cells.items() if k != row["key"] for c in cs]
        neg_nll: list[float] = []
        for tag, cell in rng.sample(others, min(n_neg_cells, len(others))):
            ids = tokenizer.encode(cell, add_special_tokens=False)
            lp, per = model.log_prob(mu, tag_ids[tag], ids)
            neg_nll.append(-lp / len(per))
        rows.append(
            {
                "key": row["key"],
                "in_nll_tok": sum(in_nll) / len(in_nll),
                "neg_nll_tok": sum(neg_nll) / len(neg_nll),
                "margin": sum(neg_nll) / len(neg_nll) - sum(in_nll) / len(in_nll),
            }
        )

    margins = sorted(r["margin"] for r in rows)
    n = len(margins)

    def pct(q: float) -> float:
        return margins[min(n - 1, int(q * n))]

    return {
        "n_companies": n,
        "mean_in_nll_tok": sum(r["in_nll_tok"] for r in rows) / n,
        "mean_neg_nll_tok": sum(r["neg_nll_tok"] for r in rows) / n,
        "margin_median": pct(0.5),
        "margin_p10": pct(0.1),
        "margin_p90": pct(0.9),
        "margin_mean": sum(margins) / n,
        "pct_positive": sum(1 for m in margins if m > 0) / n,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Selectivity probe over a key range of company clusters.")
    parser.add_argument("--config", required=True, help="Training config yaml (model + data paths).")
    parser.add_argument("--checkpoint", required=True, help="Lightning checkpoint (.ckpt).")
    parser.add_argument("--key-range", default="2000:3000", help="Test key index range [lo, hi) (v3 test companies).")
    parser.add_argument("--n-companies", type=int, default=100, help="Panel size.")
    parser.add_argument("--n-neg-cells", type=int, default=20, help="Negative cells sampled per company.")
    parser.add_argument("--seed", type=int, default=123, help="Panel/negative sampling seed.")
    parser.add_argument("--split", default="test", help="Split to read the panel from (test/val).")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out", default=None, help="Optional JSON report path.")
    args = parser.parse_args(argv)

    helpers.bootstrap_logging(logging.INFO)
    import torch

    with open(args.config, encoding="utf-8") as handle:
        root_cfg = ConfigForRoot.from_mapping(yaml.safe_load(handle))
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = _setup_tokenizer(root_cfg.model.tokenizer_path)
    model = load_model(root_cfg, tokenizer, Path(args.checkpoint))
    model.eval().to(device)
    log.info("Loaded %s on %s", args.checkpoint, device)

    lo, hi = (int(x) for x in args.key_range.split(":"))
    split_cfg = root_cfg.splits[args.split]
    pattern = f"{root_cfg.data_root}/{split_cfg.dataset.pattern}"
    records = [
        rec
        for rec in ClusterSampleAdapter(urls=pattern, seed=root_cfg.seed, loop_back=False)
        if in_range(rec["key"], lo, hi)
    ]
    log.info("%s records in key range [%d, %d): %d", args.split, lo, hi, len(records))
    if not records:
        raise SystemExit("No records in the key range.")

    stats = run_probe(
        model, tokenizer, records, n_companies=args.n_companies, n_neg_cells=args.n_neg_cells, seed=args.seed
    )
    print(f"=== selectivity: {stats['n_companies']} companies, keys {lo}..{hi - 1} ===")
    print(f"  in-cluster NLL/tok:  {stats['mean_in_nll_tok']:.3f}")
    print(f"  negative NLL/tok:    {stats['mean_neg_nll_tok']:.3f}")
    print(
        f"  margin: median {stats['margin_median']:.3f}  p10 {stats['margin_p10']:.3f}  "
        f"p90 {stats['margin_p90']:.3f}  mean {stats['margin_mean']:.3f}  positive {stats['pct_positive']:.0%}"
    )
    if args.out:
        Path(args.out).write_text(
            json.dumps(dict(stats, checkpoint=str(args.checkpoint), key_range=[lo, hi]), indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
