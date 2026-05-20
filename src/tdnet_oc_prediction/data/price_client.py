import pandas as pd
from .validators import validate_prices


class PriceClient:
    def __init__(self, source_path: str):
        self.source_path = source_path

    def fetch(self, stock_codes: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        df = pd.read_csv(self.source_path)
        if "high" not in df.columns:
            df["high"] = df["close"]
        if "low" not in df.columns:
            df["low"] = df["close"]
        if "volume" not in df.columns:
            df["volume"] = 0
        df["stock_code"] = df["stock_code"].astype(str)
        df["date"] = pd.to_datetime(df["date"])
        mask = (df["stock_code"].isin([str(s) for s in stock_codes])) & (df["date"] >= start_date) & (df["date"] <= end_date)
        return validate_prices(df.loc[mask].copy())
