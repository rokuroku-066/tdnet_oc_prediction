import yaml
from .schema import Config

def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        return Config.model_validate(yaml.safe_load(f))
