import argparse
import shutil
from pathlib import Path
from tdnet_oc_prediction.config.loader import load_config


def _copy_if_needed(src: Path, dst: Path) -> None:
    src_resolved = src.resolve()
    dst_resolved = dst.resolve()
    if src_resolved == dst_resolved:
        return
    shutil.copy2(src_resolved, dst_resolved)


def main(config_path: str):
    cfg = load_config(config_path)
    src_d = Path(cfg.data["disclosure_path"])
    src_p = Path(cfg.data["price_path"])
    out_d = Path("data/raw/disclosures") / src_d.name
    out_p = Path("data/raw/prices") / src_p.name
    out_d.parent.mkdir(parents=True, exist_ok=True)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    _copy_if_needed(src_d, out_d)
    _copy_if_needed(src_p, out_p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
