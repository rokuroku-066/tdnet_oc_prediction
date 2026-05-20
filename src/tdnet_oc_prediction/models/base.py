import numpy as np
class BaseModel:
    def fit(self, train_df, valid_df=None): raise NotImplementedError
    def predict_proba(self, df) -> np.ndarray: raise NotImplementedError
    def predict(self, df, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(df) >= threshold).astype(int)
    def save(self, path: str): raise NotImplementedError
    @classmethod
    def load(cls, path: str): raise NotImplementedError
