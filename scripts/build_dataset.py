import argparse, pandas as pd
from tdnet_oc_prediction.config.loader import load_config
from tdnet_oc_prediction.data.disclosure_client import DisclosureClient
from tdnet_oc_prediction.data.price_client import PriceClient
from tdnet_oc_prediction.data.dataset_builder import DatasetBuilder, TimeSeriesSplitter
from tdnet_oc_prediction.utils.io import save_df

def main(config_path):
    cfg = load_config(config_path)
    d = DisclosureClient(cfg.data['disclosure_path']).fetch(cfg.data['start_date'], cfg.data['end_date'])
    p = PriceClient(cfg.data['price_path']).fetch(d['stock_code'].astype(str).unique().tolist(), cfg.data['start_date'], cfg.data['end_date'])
    ds = DatasetBuilder().build(d, p)
    save_df('data/processed/dataset.parquet', ds)
    splits = TimeSeriesSplitter(cfg.split).split(ds)
    for k,v in splits.items(): save_df(f'data/splits/{k}.parquet', v)

if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--config', required=True); args=ap.parse_args(); main(args.config)
