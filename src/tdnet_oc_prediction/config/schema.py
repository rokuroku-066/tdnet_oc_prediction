from pydantic import BaseModel

class ProjectConfig(BaseModel):
    name: str = "tdnet_oc_prediction"
    seed: int = 42
    timezone: str = "Asia/Tokyo"

class Config(BaseModel):
    project: ProjectConfig = ProjectConfig()
    data: dict = {}
    split: dict = {}
    model: dict = {}
    evaluation: dict = {}
    simulation: dict = {}
