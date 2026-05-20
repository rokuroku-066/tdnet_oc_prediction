import argparse
from pathlib import Path

from tdnet_oc_prediction.config.loader import load_config
from tdnet_oc_prediction.data.disclosure_client import DisclosureClient
from tdnet_oc_prediction.data.price_client import PriceClient
from tdnet_oc_prediction.data.stooq_client import StooqClient
from tdnet_oc_prediction.data.tdnet_client import TDnetPublicClient


def main(config_path: str):
    cfg = load_config(config_path)
    data_cfg = cfg.data

    disclosure_source = data_cfg.get("disclosure_source", "csv")
    price_source = data_cfg.get("price_source", "csv")
    start_date = data_cfg["start_date"]
    end_date = data_cfg["end_date"]

    if disclosure_source == "csv":
        disclosures = DisclosureClient(data_cfg["disclosure_path"]).fetch(start_date, end_date)
    elif disclosure_source == "tdnet_public":
        disclosures = TDnetPublicClient(
            sleep_sec=float(data_cfg.get("request_sleep_sec", 0.2)),
            timeout_sec=float(data_cfg.get("request_timeout_sec", 20.0)),
            max_pages=int(data_cfg.get("tdnet_max_pages", 20)),
            extract_pdf=bool(data_cfg.get("tdnet_extract_pdf", True)),
        ).fetch(start_date, end_date)
    else:
        raise ValueError(f"unsupported disclosure_source: {disclosure_source}")

    if disclosures.empty:
        raise RuntimeError("No disclosure data collected")

    stock_codes = disclosures["stock_code"].astype(str).dropna().unique().tolist()

    if price_source == "csv":
        prices = PriceClient(data_cfg["price_path"]).fetch(stock_codes, start_date, end_date)
    elif price_source == "stooq":
        prices = StooqClient(
            sleep_sec=float(data_cfg.get("request_sleep_sec", 0.2)),
            timeout_sec=float(data_cfg.get("request_timeout_sec", 20.0)),
        ).fetch(stock_codes, start_date, end_date)
    else:
        raise ValueError(f"unsupported price_source: {price_source}")

    if prices.empty:
        raise RuntimeError("No price data collected")

    out_d = Path(data_cfg["disclosure_path"])
    out_p = Path(data_cfg["price_path"])
    out_d.parent.mkdir(parents=True, exist_ok=True)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    disclosures.to_csv(out_d, index=False)
    prices.to_csv(out_p, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
