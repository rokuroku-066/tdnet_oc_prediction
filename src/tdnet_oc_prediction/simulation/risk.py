import numpy as np

def simulation_metrics(net_returns):
    r = np.array(net_returns)
    traded = r[r != 0]
    if len(traded) == 0:
        return {'num_trades':0,'win_rate':0.0,'mean_return':0.0,'cumulative_return':0.0,'sharpe_ratio':0.0,'max_drawdown':0.0}
    eq = np.cumprod(1 + traded)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    return {
        'num_trades': int(len(traded)),
        'win_rate': float((traded > 0).mean()),
        'mean_return': float(traded.mean()),
        'cumulative_return': float(eq[-1] - 1),
        'sharpe_ratio': float((traded.mean() / traded.std()) if traded.std() > 0 else 0.0),
        'max_drawdown': float(dd.min()),
    }
