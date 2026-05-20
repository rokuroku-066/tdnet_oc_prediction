from ..models.registry import build_model

class TrainerService:
    def train(self, model_cfg: dict, train_df, valid_df):
        m = build_model(model_cfg)
        m.fit(train_df, valid_df)
        return m
