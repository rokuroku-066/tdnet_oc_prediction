import joblib
from sklearn.linear_model import LogisticRegression
from .base import BaseModel
from ..features.tfidf_vectorizer import build_vectorizer
class TfidfLogRegModel(BaseModel):
    def __init__(self, tfidf_cfg: dict, clf_cfg: dict):
        self.vectorizer=build_vectorizer(tfidf_cfg); self.classifier=LogisticRegression(**clf_cfg)
    def fit(self, train_df, valid_df=None):
        x=self.vectorizer.fit_transform(train_df['text']); self.classifier.fit(x, train_df['y'])
    def predict_proba(self, df):
        x=self.vectorizer.transform(df['text']); return self.classifier.predict_proba(x)[:,1]
    def save(self, path): joblib.dump({'vectorizer':self.vectorizer,'classifier':self.classifier}, path)
    @classmethod
    def load(cls, path):
        d=joblib.load(path); m=cls({},{}); m.vectorizer=d['vectorizer']; m.classifier=d['classifier']; return m
