import argparse
import subprocess


def run_capture(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""


def run(cmd: list[str]):
    subprocess.run(cmd, check=True)


def main(config: str):
    run(["python", "scripts/collect_data.py", "--config", config])
    run(["python", "scripts/build_dataset.py", "--config", config])
    rid = run_capture(["python", "scripts/train.py", "--config", config])
    run(["python", "scripts/evaluate.py", "--config", config, "--model-dir", f"models/{rid}"])
    run(["python", "scripts/simulate.py", "--config", config, "--predictions", f"data/predictions/{rid}_test_predictions.parquet"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
