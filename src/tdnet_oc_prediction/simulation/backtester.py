import numpy as np
import pandas as pd
from .cost import apply_cost


class Backtester:
    def run(self, signal_df: pd.DataFrame, transaction_cost_bps: float = 0.0) -> pd.DataFrame:
        df = signal_df.copy()
        df['entry_price'] = df['target_open']
        df['exit_price'] = df['target_close']
        long_ret = (df['exit_price'] - df['entry_price']) / df['entry_price']
        short_ret = (df['entry_price'] - df['exit_price']) / df['entry_price']
        df['gross_return'] = np.where(df['signal'] == 1, long_ret, np.where(df['signal'] == -1, short_ret, 0.0))
        df['net_return'] = np.where(
            df['signal'] != 0,
            df['gross_return'].apply(lambda r: apply_cost(r, transaction_cost_bps)),
            0.0,
        )
        df['trade_date'] = df['target_date']
        return df[['sample_id','stock_code','trade_date','signal','entry_price','exit_price','gross_return','net_return','y_true','y_proba']]
