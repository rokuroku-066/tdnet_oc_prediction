import pandas as pd
from .validators import validate_disclosures


class DisclosureClient:
    def __init__(self, source_path: str):
        self.source_path = source_path

    def fetch(self, start_date: str, end_date: str) -> pd.DataFrame:
        df = pd.read_csv(self.source_path)
        defaults = {
            "company_name": "",
            "disclosure_time": "",
            "source_url": "",
            "pdf_url": "",
            "title": "",
            "body_text": "",
        }
        for k, v in defaults.items():
            if k not in df.columns:
                df[k] = v
        df["stock_code"] = df["stock_code"].astype(str)
        df["disclosure_date"] = pd.to_datetime(df["disclosure_date"])
        mask = (df["disclosure_date"] >= start_date) & (df["disclosure_date"] <= end_date)
        return validate_disclosures(df.loc[mask].copy())
