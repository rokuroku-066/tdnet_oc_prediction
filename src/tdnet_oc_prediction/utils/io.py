import json, os
import pandas as pd

def ensure_dir(path: str): os.makedirs(path, exist_ok=True)
def save_json(path: str, data: dict):
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as f: json.dump(data, f, ensure_ascii=False, indent=2)
def save_df(path: str, df: pd.DataFrame):
    ensure_dir(os.path.dirname(path))
    if path.endswith('.parquet'): df.to_parquet(path, index=False)
    else: df.to_csv(path, index=False)
