import pandas as pd

def validate_disclosures(df: pd.DataFrame) -> pd.DataFrame:
    req = ["disclosure_id", "stock_code", "disclosure_date", "title", "body_text"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    df = df.dropna(subset=["stock_code", "disclosure_date"])
    df = df[(df["title"].fillna("") + df["body_text"].fillna("")).str.len() > 0]
    return df.drop_duplicates(subset=["disclosure_id"])

def validate_prices(df: pd.DataFrame) -> pd.DataFrame:
    req = ["stock_code", "date", "open", "close"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    df = df.dropna(subset=["stock_code", "date", "open", "close"])
    return df[(df["open"] > 0) & (df["close"] > 0)]
