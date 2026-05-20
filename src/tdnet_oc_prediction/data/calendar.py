import pandas as pd

def next_business_day(date: pd.Timestamp, trading_dates: pd.Series) -> pd.Timestamp | None:
    future = trading_dates[trading_dates > date]
    return future.iloc[0] if not future.empty else None
