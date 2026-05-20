import joblib, numpy as np
from .base import BaseModel
class MajorityBaseline(BaseModel):
    def __init__(self): self.majority_label=1; self.positive_rate=0.5
    def fit(self, train_df, valid_df=None):
        y=train_df['y'].astype(int); self.positive_rate=float(y.mean()); self.majority_label=int((self.positive_rate>=0.5))
    def predict_proba(self, df): return np.full(len(df), self.positive_rate)
    def predict(self, df, threshold: float=0.5): return np.full(len(df), self.majority_label)
    def save(self, path): joblib.dump({'majority_label':self.majority_label,'positive_rate':self.positive_rate}, path)
    @classmethod
    def load(cls, path):
        d=joblib.load(path); m=cls(); m.majority_label=d['majority_label']; m.positive_rate=d['positive_rate']; return m
