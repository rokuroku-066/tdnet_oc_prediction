import argparse, os, pandas as pd
from tdnet_oc_prediction.config.loader import load_config
from tdnet_oc_prediction.simulation.signal import SignalGenerator
from tdnet_oc_prediction.simulation.backtester import Backtester
from tdnet_oc_prediction.simulation.risk import simulation_metrics
from tdnet_oc_prediction.utils.io import save_df, save_json

def main(config_path, predictions):
    cfg=load_config(config_path)
    pred=pd.read_parquet(predictions)
    s=SignalGenerator().generate(pred, cfg.simulation['threshold_long'], cfg.simulation['threshold_short'], cfg.simulation.get('allow_short',True))
    t=Backtester().run(s, cfg.simulation.get('transaction_cost_bps',0))
    rid=os.path.basename(predictions).replace('_test_predictions.parquet','')
    save_df(f'data/simulations/{rid}_trades.parquet', t)
    save_json(f'reports/simulation/{rid}_simulation_metrics.json', simulation_metrics(t['net_return']))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--config', required=True); ap.add_argument('--predictions', required=True); a=ap.parse_args(); main(a.config,a.predictions)
