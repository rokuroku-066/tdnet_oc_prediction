import argparse
import json
from pathlib import Path
import pandas as pd
import yaml
from tdnet_oc_prediction.config.loader import load_config
from tdnet_oc_prediction.evaluation.evaluator import Evaluator
from tdnet_oc_prediction.training.trainer import TrainerService
from tdnet_oc_prediction.utils.io import save_json
from tdnet_oc_prediction.utils.time import run_id


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
    save_json(str(out_dir / "train_log.json"), {"run_id": rid, "train_size": len(tr), "valid_size": len(va)})
    print(rid)
    return rid


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
