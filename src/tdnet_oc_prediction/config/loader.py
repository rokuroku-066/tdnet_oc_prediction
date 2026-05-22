import yaml
from pydantic import ValidationError

from .schema import Config


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}

    try:
        return Config.model_validate(payload)
    except ValidationError as exc:
        first_error = exc.errors()[0] if exc.errors() else {}
        loc = first_error.get("loc", ())
        section = loc[0] if loc else "root"
        field_path = ".".join(str(item) for item in loc) if loc else "root"
        message = first_error.get("msg", str(exc))
        raise ValueError(
            f"Invalid config section '{section}' at '{field_path}': {message}"
        ) from exc
