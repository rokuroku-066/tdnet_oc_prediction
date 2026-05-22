import argparse
from datetime import datetime
import logging
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo
import pandas as pd
import yaml
from tdnet_oc_prediction.config.loader import load_config
from tdnet_oc_prediction.evaluation.evaluator import Evaluator
from tdnet_oc_prediction.training.trainer import TrainerService
from tdnet_oc_prediction.utils.io import save_json
from tdnet_oc_prediction.utils.time import run_id

LOGGER = logging.getLogger(__name__)


def _get_git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except subprocess.CalledProcessError as exc:
        LOGGER.warning("failed to get git commit hash due to git command error: %s", exc)
        return "UNAVAILABLE"
    except FileNotFoundError as exc:
        LOGGER.warning("failed to get git commit hash because git command was not found: %s", exc)
        return "UNAVAILABLE"


def _get_run_timestamp(timezone_name: str) -> str:
    tz = ZoneInfo(timezone_name) if timezone_name else ZoneInfo("UTC")
    return datetime.now(tz).isoformat()


def _label_distribution(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    counts = df["y"].value_counts(dropna=False).to_dict()
    total = len(df)
    distribution: dict[str, dict[str, float]] = {}
    for label, count in counts.items():
        key = "null" if pd.isna(label) else str(label)
        ratio = (count / total) if total else 0.0
        distribution[key] = {"count": int(count), "ratio": float(ratio)}
    return distribution


def main(config_path: str) -> str:
    cfg = load_config(config_path)
    tr = pd.read_parquet("data/splits/train.parquet")
    va = pd.read_parquet("data/splits/valid.parquet")
    model = TrainerService().train(cfg.model, tr, va)
    rid = run_id(cfg.model["name"])
    out_dir = Path("models") / rid
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir / "model.pkl") if cfg.model["name"] != "transformer" else str(out_dir))
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg.model, allow_unicode=True), encoding="utf-8")
    valid_proba = model.predict_proba(va)
    valid_metrics = Evaluator().evaluate(va["y"].values, valid_proba, cfg.model.get("threshold", 0.5))
    save_json(str(out_dir / "metrics_valid.json"), valid_metrics)

    log_payload = {
        "run_id": rid,
        "executed_at": _get_run_timestamp(cfg.project.timezone),
        "git_commit": _get_git_commit_hash(),
        "train_size": len(tr),
        "valid_size": len(va),
        "label_distribution": {
            "train": _label_distribution(tr),
            "valid": _label_distribution(va),
        },
        "config_summary": {
            "model": cfg.model,
            "split": cfg.split,
            "seed": cfg.project.seed,
        },
    }
    save_json(str(out_dir / "train_log.json"), log_payload)
    print(rid)
    return rid


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
