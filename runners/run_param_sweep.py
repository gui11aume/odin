"""Parameter sweep for Odin VAE training.

Runs a grid of hyperparameter combos (learning rate, batch size, gradient
accumulation, ...) as separate single-GPU training processes and aggregates
the per-epoch validation metrics into a results CSV.

Each combo trains for ``--epochs`` epochs (default 1 for a first pass) on one
GPU; up to ``len(--gpus)`` combos run in parallel, each pinned to its own GPU
via CUDA_VISIBLE_DEVICES. Every combo gets a unique logger name, so its log
dir is ``lightning_logs/<log_name>`` and can be found unambiguously even with
concurrent jobs. Checkpointing is disabled for sweep runs (metrics only);
retrain the winner with checkpointing enabled.

Usage:
    uv run python runners/run_param_sweep.py \
        --base runners/odin_vae_config.yaml \
        --combos "lr=2e-4,batch=256,accum=1;lr=5e-4,batch=256,accum=1;lr=2e-4,batch=256,accum=4" \
        --gpus 1,2 --epochs 1 \
        [--shards 8] [--tag sweep1]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import yaml

RUNNER_DIR = Path(__file__).resolve().parent
if str(RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(RUNNER_DIR))

import helpers  # noqa: E402

log = logging.getLogger(__name__)

KNOWN_KEYS = ("lr", "batch", "accum", "epochs", "warmup", "wd", "kl")
FIELDNAMES = [
    "combo",
    "lr",
    "batch",
    "accum",
    "effective_batch",
    "warmup",
    "wd",
    "kl",
    "epochs",
    "gpu",
    "final_val_loss",
    "final_val_ce",
    "final_val_kl",
    "final_train_loss",
    "duration_s",
    "log_dir",
    "status",
]


def parse_combos(spec: str) -> list[dict[str, float]]:
    """``"lr=1e-3,batch=256;lr=5e-4,batch=128,accum=2"`` -> list of override dicts."""
    combos: list[dict[str, float]] = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        combo: dict[str, float] = {}
        for part in chunk.split(","):
            key, _, value = part.strip().partition("=")
            key = key.strip()
            if key not in KNOWN_KEYS:
                raise ValueError(f"Unknown sweep key {key!r} (expected one of {', '.join(KNOWN_KEYS)}).")
            combo[key] = float(value)
        if not combo:
            raise ValueError("Empty combo in sweep spec.")
        combos.append(combo)
    return combos


def combo_slug(combo: dict[str, float]) -> str:
    return "-".join(f"{k}{v:g}" for k, v in sorted(combo.items()))


def build_combo_config(base: dict, combo: dict[str, float], *, tag: str, epochs: int, shards: int | None) -> dict:
    """Overlay the sweep overrides onto a deep copy of the base config."""
    cfg = yaml.safe_load(yaml.safe_dump(base))  # deep copy via yaml round-trip
    training = cfg["training"]
    training["lr"] = combo.get("lr", training["lr"])
    training["max_epochs"] = int(combo.get("epochs", epochs))
    training["accumulate_grad_batches"] = int(combo.get("accum", 1))
    if "warmup" in combo:
        training["lr_warmup_ratio"] = combo["warmup"]

    if "wd" in combo:
        training["optimizer_kwargs"]["weight_decay"] = combo["wd"]
    if "kl" in combo:
        cfg["model"]["kl_weight"] = combo["kl"]
    training["enable_checkpointing"] = False
    training["devices"] = 1
    training["strategy"] = "auto"
    training["log_name"] = f"{tag}_{combo_slug(combo)}"
    if "batch" in combo:
        cfg["splits"]["train"]["dataloader"]["batch_size"] = int(combo["batch"])
    if shards is not None:
        cfg["splits"]["train"]["dataset"]["pattern"] = f"train/shard-{{000000..{shards - 1:06d}}}.tar.gz"
        # The entry point's auto limit reads the full-corpus manifest, so a
        # sharded smoke run needs an explicit epoch length or the endless
        # stream recycles the shards until the full-corpus batch count.
        batch = cfg["splits"]["train"]["dataloader"]["batch_size"]
        n_instances = shards * int(cfg["splits"]["train"]["dataset"]["n_instances_per_shard"])
        cfg["training"]["limit_train_batches"] = max(1, (n_instances + batch - 1) // batch)
    return cfg


def read_metrics(log_name: str, repo_root: Path) -> dict:
    """Final + per-epoch val metrics from a combo's CSVLogger file.

    The entry point pre-creates the log dir (config snapshot), so CSVLogger
    nests its output in a ``version_0`` sub-directory; glob covers both.
    """
    candidates = sorted((repo_root / "lightning_logs" / log_name).glob("version_*/metrics.csv"))
    direct = repo_root / "lightning_logs" / log_name / "metrics.csv"
    metrics_csv = candidates[-1] if candidates else direct
    with open(metrics_csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    history: list[dict] = []
    for row in rows:
        if row.get("val_loss"):
            history.append(
                {
                    "epoch": int(row["epoch"]),
                    "val_loss": float(row["val_loss"]),
                    "val_ce": float(row["val_ce"]) if row.get("val_ce") else None,
                    "val_kl": float(row["val_kl"]) if row.get("val_kl") else None,
                    "train_loss": float(row["train_loss"]) if row.get("train_loss") else None,
                }
            )
    if not history:
        raise ValueError(f"No val_loss rows in {metrics_csv}")
    # train_epoch metrics land in their own row; take the last one available.
    train_rows = [row for row in rows if row.get("train_loss")]
    if train_rows:
        history[-1]["train_loss"] = float(train_rows[-1]["train_loss"])
    return {"final": history[-1], "history": history}


def run_combo(cfg_path: Path, gpu: int, log_path: Path, repo_root: Path) -> int:
    env = {
        **subprocess.os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            [sys.executable, str(RUNNER_DIR / "run_train_odin_vae.py"), "--config", str(cfg_path)],
            cwd=repo_root,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    return proc.returncode


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sweep Odin VAE training hyperparameters.")
    parser.add_argument("--base", required=True, help="Base training config yaml.")
    parser.add_argument("--combos", required=True, help="Semicolon-separated combo list (see module docstring).")
    parser.add_argument("--gpus", default="1,2", help="Comma-separated GPU ids for parallel jobs.")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs per combo (default 1 for a first pass).")
    parser.add_argument(
        "--shards", type=int, default=None, help="Limit the train pattern to the first N shards (smoke)."
    )
    parser.add_argument("--tag", default="sweep", help="Run tag for combo logger names.")
    parser.add_argument("--sweep-dir", default=None, help="Output dir (default lightning_logs/sweep_<tag>).")
    args = parser.parse_args(argv)

    helpers.bootstrap_logging(logging.INFO)
    repo_root = Path(__file__).resolve().parents[1]
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    combos = parse_combos(args.combos)
    base = yaml.safe_load(Path(args.base).read_text(encoding="utf-8"))
    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else repo_root / "lightning_logs" / f"sweep_{args.tag}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    out_csv = sweep_dir / f"{args.tag}_results.csv"

    with open(out_csv, "a", newline="", encoding="utf-8") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=FIELDNAMES)
        if out_csv.stat().st_size == 0:
            writer.writeheader()
        pending = list(enumerate(combos))
        running: dict[subprocess.Popen, dict] = {}
        free_gpus = set(gpus)
        while pending or running:
            while pending and free_gpus:
                i, combo = pending.pop(0)
                gpu = sorted(free_gpus)[0]
                free_gpus.discard(gpu)
                slug = combo_slug(combo)
                combo_dir = sweep_dir / f"combo_{i:02d}_{slug}"
                combo_dir.mkdir(parents=True, exist_ok=True)
                cfg = build_combo_config(base, combo, tag=f"{args.tag}_{i:02d}", epochs=args.epochs, shards=args.shards)
                cfg_path = combo_dir / "config.yaml"
                cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
                log.info(
                    "launch [%s] lr=%s batch=%s accum=%s epochs=%s gpu=%s",
                    slug,
                    cfg["training"]["lr"],
                    cfg["splits"]["train"]["dataloader"]["batch_size"],
                    cfg["training"]["accumulate_grad_batches"],
                    cfg["training"]["max_epochs"],
                    gpu,
                )
                proc = subprocess.Popen(
                    [sys.executable, str(RUNNER_DIR / "run_train_odin_vae.py"), "--config", str(cfg_path)],
                    cwd=repo_root,
                    env={
                        **subprocess.os.environ,
                        "CUDA_VISIBLE_DEVICES": gpu,
                        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                    },
                    stdout=open(combo_dir / "train.log", "w", encoding="utf-8"),
                    stderr=subprocess.STDOUT,
                )
                started = time.time()
                running[proc] = {
                    "combo": combo,
                    "slug": slug,
                    "gpu": gpu,
                    "cfg": cfg,
                    "started": started,
                    "log_name": cfg["training"]["log_name"],
                    "combo_dir": combo_dir,
                }
            for proc in [p for p in running if p.poll() is not None]:
                info = running.pop(proc)
                duration = time.time() - info["started"]
                free_gpus.add(info["gpu"])
                row = {
                    "combo": info["slug"],
                    "lr": info["cfg"]["training"]["lr"],
                    "batch": info["cfg"]["splits"]["train"]["dataloader"]["batch_size"],
                    "accum": info["cfg"]["training"]["accumulate_grad_batches"],
                    "effective_batch": info["cfg"]["splits"]["train"]["dataloader"]["batch_size"]
                    * info["cfg"]["training"]["accumulate_grad_batches"],
                    "warmup": info["cfg"]["training"].get("lr_warmup_ratio"),
                    "wd": info["cfg"]["training"]["optimizer_kwargs"].get("weight_decay"),
                    "kl": info["cfg"]["model"].get("kl_weight"),
                    "epochs": info["cfg"]["training"]["max_epochs"],
                    "gpu": info["gpu"],
                    "duration_s": round(duration, 1),
                    "log_dir": f"lightning_logs/{info['log_name']}",
                }
                if proc.returncode != 0:
                    row["status"] = f"failed(rc={proc.returncode})"
                else:
                    try:
                        metrics = read_metrics(info["log_name"], repo_root)
                        row.update(
                            {
                                "final_val_loss": metrics["final"]["val_loss"],
                                "final_val_ce": metrics["final"]["val_ce"],
                                "final_val_kl": metrics["final"]["val_kl"],
                                "final_train_loss": metrics["final"]["train_loss"],
                            }
                        )
                        row["status"] = "ok"
                        (info["combo_dir"] / "val_history.json").write_text(
                            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
                        )
                    except (ValueError, OSError, KeyError) as exc:
                        row["status"] = f"metrics-error({exc})"
                writer.writerow(row)
                csv_handle.flush()
                log.info(
                    "done   [%s] gpu=%s %s val_loss=%s",
                    info["slug"],
                    info["gpu"],
                    row["status"],
                    row.get("final_val_loss"),
                )
            time.sleep(5)
    log.info("sweep complete: %s", out_csv)


if __name__ == "__main__":
    main()
