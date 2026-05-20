import pandas as pd
from .validators import validate_prices

class PriceClient:
    def __init__(self, source_path: str):
        self.source_path = source_path

    def fetch(self, stock_codes: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        df = pd.read_csv(self.source_path)
        df["date"] = pd.to_datetime(df["date"])
        mask = (df["stock_code"].astype(str).isin(stock_codes)) & (df["date"] >= start_date) & (df["date"] <= end_date)
        return validate_prices(df.loc[mask].copy())
