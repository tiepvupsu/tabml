from tabml.schemas.feature_config import FeatureConfig
from tabml.schemas.pipeline_config import PipelineConfig
import pydantic
from typing import Dict, Any


class FeatureBundle(pydantic.BaseModel):
    feature_config: FeatureConfig
    transformers: Dict[str, Any]


class ModelBundle(pydantic.BaseModel):
    pipeline_config: PipelineConfig
    model: Any


class FullPipelineBundle(pydantic.BaseModel):
    feature_bundle: FeatureBundle
    model_bundle: ModelBundle
