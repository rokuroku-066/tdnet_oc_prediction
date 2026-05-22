import numpy as np
import pandas as pd
from .calendar import next_business_day

def make_label(open_price: float, close_price: float):
    if close_price > open_price:
        return 1
    if close_price < open_price:
        return 0
    return None

class DatasetBuilder:
    def build(self, disclosures: pd.DataFrame, prices: pd.DataFrame, calendar: pd.DataFrame | None = None) -> pd.DataFrame:
        d = disclosures.copy()
        d["disclosure_date"] = pd.to_datetime(d["disclosure_date"]).dt.normalize()
        d["text_piece"] = "[" + d["title"] + "]\n" + d["body_text"]
        agg = d.groupby(["stock_code", "disclosure_date"], as_index=False).agg(
            text=("text_piece", "\n\n".join),
            num_disclosures=("disclosure_id", "count"),
        )
        if "text" not in agg.columns:
            raise ValueError("Aggregation failed to create required 'text' column")
        prices = prices.copy()
        prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
        if calendar is not None:
            cal = calendar.copy()
            required_calendar_cols = {"date"}
            missing_calendar_cols = required_calendar_cols - set(cal.columns)
            if missing_calendar_cols:
                raise ValueError(f"calendar is missing required columns: {sorted(missing_calendar_cols)}")
            cal["date"] = pd.to_datetime(cal["date"]).dt.normalize()
            trading = cal["date"].drop_duplicates().sort_values().reset_index(drop=True)
        else:
            trading = prices["date"].drop_duplicates().sort_values().reset_index(drop=True)
        agg["target_date"] = agg["disclosure_date"].apply(lambda x: next_business_day(x, trading))
        merged = agg.merge(
            prices[["stock_code", "date", "open", "close"]],
            left_on=["stock_code", "target_date"],
            right_on=["stock_code", "date"],
            how="left",
        )
        merged = merged.rename(columns={"open": "target_open", "close": "target_close"}).drop(columns=["date"])
        merged["y"] = np.select(
            [merged["target_close"] > merged["target_open"], merged["target_close"] < merged["target_open"]],
            [1, 0],
            default=np.nan,
        )
        merged = merged.dropna(subset=["target_date", "target_open", "target_close", "y"]).copy()
        merged["y"] = merged["y"].astype(int)
        merged["sample_id"] = merged["stock_code"].astype(str) + "_" + merged[
            "disclosure_date"
        ].dt.strftime("%Y-%m-%d")

        output_columns = [
            "sample_id",
            "stock_code",
            "disclosure_date",
            "target_date",
            "text",
            "target_open",
            "target_close",
            "y",
            "num_disclosures",
        ]
        missing_output_columns = sorted(set(output_columns) - set(merged.columns))
        if missing_output_columns:
            raise ValueError(
                f"dataset is missing required output columns: {missing_output_columns}"
            )
        return merged[output_columns]

class TimeSeriesSplitter:
    def __init__(self, split_conf: dict):
        self.c = split_conf

    def split(self, dataset: pd.DataFrame) -> dict[str, pd.DataFrame]:
        train_end = pd.to_datetime(self.c["train_end"])
        valid_start = pd.to_datetime(self.c["valid_start"])
        valid_end = pd.to_datetime(self.c["valid_end"])
        test_start = pd.to_datetime(self.c["test_start"])
        test_end = pd.to_datetime(self.c["test_end"])

        if not (train_end < valid_start <= valid_end < test_start <= test_end):
            raise ValueError("Invalid split_conf: train/valid/test date ranges overlap or are out of order")

        if not (test_start > train_end):
            raise ValueError("Invalid split_conf: test period must be in the future of train period")

        d = dataset.copy()
        d["disclosure_date"] = pd.to_datetime(d["disclosure_date"])
        tr = d[d["disclosure_date"] <= train_end]
        va = d[(d["disclosure_date"] >= valid_start) & (d["disclosure_date"] <= valid_end)]
        te = d[(d["disclosure_date"] >= test_start) & (d["disclosure_date"] <= test_end)]

        train_dates = set(tr["disclosure_date"])
        valid_dates = set(va["disclosure_date"])
        test_dates = set(te["disclosure_date"])
        if train_dates & valid_dates or train_dates & test_dates or valid_dates & test_dates:
            raise ValueError("Split result is invalid: disclosure_date sets must be mutually exclusive")

        return {"train": tr, "valid": va, "test": te}
