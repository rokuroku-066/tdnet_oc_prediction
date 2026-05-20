import argparse, os, pandas as pd
import matplotlib.pyplot as plt
from tdnet_oc_prediction.config.loader import load_config
from tdnet_oc_prediction.simulation.signal import SignalGenerator
from tdnet_oc_prediction.simulation.backtester import Backtester
from tdnet_oc_prediction.simulation.risk import equity_series, simulation_metrics
from tdnet_oc_prediction.utils.io import save_df, save_json

def main(config_path, predictions):
    cfg=load_config(config_path)
    pred=pd.read_parquet(predictions)
    s=SignalGenerator().generate(pred, cfg.simulation['threshold_long'], cfg.simulation['threshold_short'], cfg.simulation.get('allow_short',True))
    t=Backtester().run(s, cfg.simulation.get('transaction_cost_bps',0))
    rid=os.path.basename(predictions).replace('_test_predictions.parquet','')
    os.makedirs('reports/simulation', exist_ok=True)
    save_df(f'data/simulations/{rid}_trades.parquet', t)
    save_json(f'reports/simulation/{rid}_simulation_metrics.json', simulation_metrics(t['net_return']))

    eq = equity_series(t['net_return'])
    plt.figure(figsize=(10, 4))
    if len(t) == 0:
        plt.axhline(1.0, color='gray', linestyle='--', linewidth=1, label='Baseline')
        plt.xlim(0, 1)
        plt.ylim(0.95, 1.05)
    else:
        plt.plot(range(len(eq)), eq, label='Equity', linewidth=1.5)
        if (t['net_return'] != 0).sum() == 0:
            plt.axhline(1.0, color='gray', linestyle='--', linewidth=1, label='Baseline')
    plt.title(f'Equity Curve: {rid}')
    plt.xlabel('Step')
    plt.ylabel('Equity')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'reports/simulation/{rid}_equity_curve.png', dpi=150)
    plt.close()

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--config', required=True); ap.add_argument('--predictions', required=True); a=ap.parse_args(); main(a.config,a.predictions)
