import argparse
import os
from pathlib import Path
import pandas as pd
from tdnet_oc_prediction.config.loader import load_config
from tdnet_oc_prediction.evaluation.evaluator import Evaluator
from tdnet_oc_prediction.models.majority import MajorityBaseline
from tdnet_oc_prediction.models.tfidf_logreg import TfidfLogRegModel
from tdnet_oc_prediction.models.transformer_classifier import TransformerClassifier
from tdnet_oc_prediction.utils.io import save_df, save_json


def load_model(model_name: str, model_dir: str):
    if model_name == "majority":
        return MajorityBaseline.load(model_dir)
    if model_name == "tfidf_logreg":
        return TfidfLogRegModel.load(model_dir)
    if model_name == "transformer":
        return TransformerClassifier.load(model_dir)
    raise ValueError(model_name)


def main(config_path: str, model_dir: str):
    cfg = load_config(config_path)
    te = pd.read_parquet("data/splits/test.parquet")
    path = model_dir if cfg.model["name"] == "transformer" else os.path.join(model_dir, "model.pkl")
    model = load_model(cfg.model["name"], path)
    proba = model.predict_proba(te)
    pred = (proba >= cfg.model.get("threshold", 0.5)).astype(int)
    out = te[["sample_id", "stock_code", "disclosure_date", "target_date", "target_open", "target_close"]].copy()
    out["y_true"] = te["y"]
    out["y_proba"] = proba
    out["y_pred"] = pred
    rid = os.path.basename(model_dir)
    save_df(f"data/predictions/{rid}_test_predictions.parquet", out)
    metrics = Evaluator().evaluate(out["y_true"].values, out["y_proba"].values, cfg.model.get("threshold", 0.5))
    save_json(f"reports/metrics/{rid}_metrics.json", metrics)
    comp_path = Path("reports/metrics/model_comparison.csv")
    row = pd.DataFrame([{"model": cfg.model["name"], **{k: metrics[k] for k in ["accuracy", "balanced_accuracy", "roc_auc", "log_loss"]}}])
    if comp_path.exists():
        all_rows = pd.concat([pd.read_csv(comp_path), row], ignore_index=True)
        all_rows = all_rows.drop_duplicates(subset=["model"], keep="last")
    else:
        all_rows = row
    all_rows.to_csv(comp_path, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-dir", required=True)
    args = ap.parse_args()
    main(args.config, args.model_dir)
