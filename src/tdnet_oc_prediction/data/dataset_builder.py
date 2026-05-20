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
        merged["y"] = merged.apply(lambda r: make_label(r["target_open"], r["target_close"]) if pd.notna(r["target_open"]) and pd.notna(r["target_close"]) else None, axis=1)
        merged = merged.dropna(subset=["target_date", "target_open", "target_close", "y"]).copy()
        merged["y"] = merged["y"].astype(int)
        merged["sample_id"] = merged.apply(lambda r: f"{r['stock_code']}_{r['disclosure_date'].date()}", axis=1)
        return merged[["sample_id","stock_code","disclosure_date","target_date","text","target_open","target_close","y","num_disclosures"]]

class TimeSeriesSplitter:
    def __init__(self, split_conf: dict): self.c = split_conf
    def split(self, dataset: pd.DataFrame) -> dict[str, pd.DataFrame]:
        d = dataset.copy(); d["disclosure_date"] = pd.to_datetime(d["disclosure_date"])
        tr = d[d["disclosure_date"] <= self.c["train_end"]]
        va = d[(d["disclosure_date"] >= self.c["valid_start"]) & (d["disclosure_date"] <= self.c["valid_end"])]
        te = d[(d["disclosure_date"] >= self.c["test_start"]) & (d["disclosure_date"] <= self.c["test_end"])]
        return {"train": tr, "valid": va, "test": te}
