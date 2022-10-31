from tabml.schemas.feature_config import FeatureConfig
from tabml.schemas.pipeline_config import PipelineConfig
import pydantic
from typing import Dict, Any


class FullPipelineData(pydantic.BaseModel):
    feature_config: FeatureConfig
    transformers: Dict[str, Any]
    pipeline_config: PipelineConfig
    model: Any
