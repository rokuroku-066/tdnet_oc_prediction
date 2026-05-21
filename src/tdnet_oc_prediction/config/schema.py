from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    name: str = "tdnet_oc_prediction"
    seed: int = 42
    timezone: str = "Asia/Tokyo"


class Config(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    data: dict = Field(default_factory=dict)
    split: dict = Field(default_factory=dict)
    model: dict = Field(default_factory=dict)
    evaluation: dict = Field(default_factory=dict)
    simulation: dict = Field(default_factory=dict)
