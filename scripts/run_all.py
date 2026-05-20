import argparse
import subprocess
from pathlib import Path


def run_capture(cmd: list[str], step_name: str) -> str:
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"[{step_name}] failed (exit={e.returncode}).\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout:\n{e.stdout}\n"
            f"stderr:\n{e.stderr}"
        ) from e
    output = proc.stdout.strip()
    return output.splitlines()[-1] if output else ""


def run(cmd: list[str], step_name: str) -> None:
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"[{step_name}] failed (exit={e.returncode}): {' '.join(cmd)}") from e


def require_path(path: Path, step_name: str, description: str) -> None:
    if not path.exists():
        raise RuntimeError(f"[{step_name}] expected {description} was not generated: {path}")


def main(config: str):
    # 仕様9.6: collect_data -> build_dataset -> train -> evaluate -> simulate
    run(["python", "scripts/collect_data.py", "--config", config], step_name="collect_data")

    run(["python", "scripts/build_dataset.py", "--config", config], step_name="build_dataset")
    for split in ["train", "valid", "test"]:
        require_path(Path(f"data/splits/{split}.parquet"), "build_dataset", f"{split} split")

    rid = run_capture(["python", "scripts/train.py", "--config", config], step_name="train")
    if not rid:
        raise RuntimeError("[train] run_id is empty; cannot continue to evaluate/simulate")
    model_dir = Path("models") / rid
    require_path(model_dir, "train", "model directory")

    run(
        ["python", "scripts/evaluate.py", "--config", config, "--model-dir", str(model_dir)],
        step_name="evaluate",
    )
    predictions_path = Path(f"data/predictions/{rid}_test_predictions.parquet")
    require_path(predictions_path, "evaluate", "test predictions parquet")

    run(
        ["python", "scripts/simulate.py", "--config", config, "--predictions", str(predictions_path)],
        step_name="simulate",
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
