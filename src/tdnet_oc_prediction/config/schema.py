from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictConfigModel(BaseModel):
    """Base config model with strict key validation and dict-like helpers."""

    model_config = ConfigDict(extra="forbid")

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class ProjectConfig(StrictConfigModel):
    name: str = "tdnet_oc_prediction"
    seed: int = 42
    timezone: str = "Asia/Tokyo"


class DataConfig(StrictConfigModel):
    disclosure_source: str = "csv"
    price_source: str = "csv"
    disclosure_path: str | None = None
    price_path: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    price_end_date: date | None = None
    price_extra_days: int = Field(default=10, ge=0, le=365)
    request_sleep_sec: float = Field(default=0.2, ge=0.0, le=60.0)
    request_timeout_sec: float = Field(default=20.0, gt=0.0, le=300.0)
    tdnet_max_pages: int = Field(default=20, ge=1, le=500)
    tdnet_extract_pdf: bool = True


class SplitConfig(StrictConfigModel):
    method: str = "time_series"
    train_end: date
    valid_start: date
    valid_end: date
    test_start: date
    test_end: date


class ModelConfig(StrictConfigModel):
    name: str = Field(default="tfidf_logreg", min_length=1)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    tfidf: dict[str, Any] = Field(default_factory=dict)
    classifier: dict[str, Any] = Field(default_factory=dict)
    pretrained_model_name: str | None = None
    max_length: int = Field(default=512, ge=8, le=8192)
    batch_size: int = Field(default=16, ge=1, le=2048)
    learning_rate: float = Field(default=2e-5, gt=0.0, le=1.0)
    epochs: float = Field(default=3, gt=0.0, le=1000)
    weight_decay: float = Field(default=0.01, ge=0.0, le=10.0)
    early_stopping_patience: int = Field(default=2, ge=0, le=100)


class EvaluationConfig(StrictConfigModel):
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class SimulationConfig(StrictConfigModel):
    threshold_long: float = Field(default=0.55, ge=0.0, le=1.0)
    threshold_short: float = Field(default=0.45, ge=0.0, le=1.0)
    allow_short: bool = True
    transaction_cost_bps: float = Field(default=10.0, ge=0.0, le=10000.0)

    @model_validator(mode="after")
    def validate_threshold_order(self) -> "SimulationConfig":
        if self.threshold_short > self.threshold_long:
            raise ValueError("threshold_short must be less than or equal to threshold_long")
        return self


class Config(StrictConfigModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    split: SplitConfig | None = None
    model: ModelConfig = Field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
