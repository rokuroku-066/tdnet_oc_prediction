import pandas as pd


def _clean_stock_code_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def validate_disclosures(df: pd.DataFrame) -> pd.DataFrame:
    req = [
        "disclosure_id",
        "stock_code",
        "company_name",
        "disclosure_date",
        "disclosure_time",
        "title",
        "body_text",
        "source_url",
        "pdf_url",
    ]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    d = df.copy()
    d["title"] = d["title"].fillna("")
    d["body_text"] = d["body_text"].fillna("")
    d = d.dropna(subset=["stock_code", "disclosure_date"]).copy()
    d["stock_code"] = _clean_stock_code_series(d["stock_code"])
    d = d[d["stock_code"].ne("")]
    d = d[d["stock_code"].str.lower().ne("nan")]
    d["disclosure_date"] = pd.to_datetime(d["disclosure_date"], errors="coerce").dt.normalize()
    d = d.dropna(subset=["disclosure_date"])
    d = d[(d["title"] + d["body_text"]).str.len() > 0]
    return d.drop_duplicates(subset=["disclosure_id"])


def validate_prices(df: pd.DataFrame) -> pd.DataFrame:
    req = ["stock_code", "date", "open", "high", "low", "close", "volume"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    d = df.copy()
    d = d.dropna(subset=["stock_code", "date", "open", "close"]).copy()
    d["stock_code"] = _clean_stock_code_series(d["stock_code"])
    d = d[d["stock_code"].ne("")]
    d = d[d["stock_code"].str.lower().ne("nan")]
    d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.normalize()
    d = d.dropna(subset=["date"])
    d[["open", "high", "low", "close", "volume"]] = d[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    d = d.dropna(subset=["open", "high", "low", "close", "volume"])
    return d[(d["open"] > 0) & (d["close"] > 0)]
