from __future__ import annotations

import io
import logging
import time

import pandas as pd
import requests

from .validators import validate_prices

LOGGER = logging.getLogger(__name__)


class StooqClient:
    def __init__(self, sleep_sec: float = 0.2, timeout_sec: float = 20.0):
        self.sleep_sec = sleep_sec
        self.timeout_sec = timeout_sec
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "tdnet-oc-prediction-research/0.1"})

    def _url(self, stock_code: str, start_date: str, end_date: str) -> str:
        d1 = pd.to_datetime(start_date).strftime("%Y%m%d")
        d2 = pd.to_datetime(end_date).strftime("%Y%m%d")
        return f"https://stooq.com/q/d/l/?s={stock_code}.jp&d1={d1}&d2={d2}&i=d"

    def fetch(self, stock_codes: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        rows: list[pd.DataFrame] = []
        for code in sorted({str(c) for c in stock_codes}):
            url = self._url(code, start_date, end_date)
            try:
                resp = self.session.get(url, timeout=self.timeout_sec)
                resp.raise_for_status()
                text = resp.text.strip()
                lower_text = text.lower()
                if not text or "no data" in lower_text:
                    LOGGER.warning("no data from stooq for %s", code)
                    continue
                df = pd.read_csv(io.StringIO(text))
                if df.empty:
                    LOGGER.warning("empty csv from stooq for %s", code)
                    continue
                rename_map = {"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
                df = df.rename(columns=rename_map)
                for col in ["date", "open", "high", "low", "close", "volume"]:
                    if col not in df.columns:
                        raise ValueError(f"missing column {col}")
                df["stock_code"] = str(code)
                rows.append(df[["stock_code", "date", "open", "high", "low", "close", "volume"]])
            except requests.RequestException as exc:
                LOGGER.warning("failed to fetch stooq data for %s: %s", code, exc)
            except (pd.errors.ParserError, ValueError, KeyError) as exc:
                LOGGER.warning("failed to parse stooq response for %s: %s", code, exc)
            time.sleep(self.sleep_sec)

        out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["stock_code", "date", "open", "high", "low", "close", "volume"])
        if out.empty:
            return out
        return validate_prices(out)
