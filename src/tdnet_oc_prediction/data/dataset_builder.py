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
        d["text_piece"] = d.apply(lambda r: f"[{r['title']}]\n{r['body_text']}", axis=1)
        agg = d.groupby(["stock_code", "disclosure_date"], as_index=False).agg(
            text=("text_piece", "\n\n".join), num_disclosures=("disclosure_id", "count")
        )
        prices = prices.copy()
        prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
        trading = prices["date"].drop_duplicates().sort_values().reset_index(drop=True)
        agg["target_date"] = agg["disclosure_date"].apply(lambda x: next_business_day(x, trading))
        merged = agg.merge(
            prices[["stock_code", "date", "open", "close"]],
            left_on=["stock_code", "target_date"],
            right_on=["stock_code", "date"],
            how="left",
        )
        merged = merged.rename(columns={"open": "target_open", "close": "target_close"}).drop(columns=["date"])
        merged["y"] = merged.apply(lambda r: make_label(r["target_open"], r["target_close"]) if pd.notna(r["target_open"]) and pd.notna(r["target_close"]) else None, axis=1)
        merged = merged.dropna(subset=["target_date", "target_open", "target_close", "y"]).copy()
        merged["y"] = merged["y"].astype(int)
        merged["sample_id"] = merged.apply(lambda r: f"{r['stock_code']}_{r['disclosure_date'].date()}", axis=1)
        return merged[["sample_id","stock_code","disclosure_date","target_date","text","target_open","target_close","y","num_disclosures"]]

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
