from pathlib import Path

import pandas as pd

from .validators import validate_prices


class PriceClient:
    def __init__(self, source_path: str):
        self.source_path = source_path

    def _build_fallback_prices(
        self,
        stock_codes: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Build minimal fallback prices when source CSV is unavailable."""
        business_days = pd.date_range(start=start_date, end=end_date, freq="B")
        rows = []
        for stock_code in stock_codes:
            for idx, price_date in enumerate(business_days):
                base = 100.0 + (idx % 20)
                close = base + 1.0 if idx % 2 == 0 else base - 1.0
                rows.append(
                    {
                        "stock_code": str(stock_code),
                        "date": price_date.date().isoformat(),
                        "open": base,
                        "high": max(base, close),
                        "low": min(base, close),
                        "close": close,
                        "volume": 1000 + idx,
                    }
                )
        return pd.DataFrame(rows)

    def fetch(self, stock_codes: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        source = Path(self.source_path)
        if source.exists():
            df = pd.read_csv(source)
        else:
            df = self._build_fallback_prices(stock_codes, start_date, end_date)

        if "high" not in df.columns:
            df["high"] = df["close"]
        if "low" not in df.columns:
            df["low"] = df["close"]
        if "volume" not in df.columns:
            df["volume"] = 0
        df["stock_code"] = df["stock_code"].astype(str)
        df["date"] = pd.to_datetime(df["date"])
        mask = (
            (df["stock_code"].isin([str(s) for s in stock_codes]))
            & (df["date"] >= pd.Timestamp(start_date))
            & (df["date"] <= pd.Timestamp(end_date))
        )
        return validate_prices(df.loc[mask].copy())
