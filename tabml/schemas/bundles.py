from typing import Any, Dict

import pydantic

from tabml.schemas.feature_config import FeatureConfig
from tabml.schemas.pipeline_config import ModelBundle


class FeatureBundle(pydantic.BaseModel):
    feature_config: FeatureConfig
    transformers: Dict[str, Any]


class PipelineBundle(pydantic.BaseModel):
    feature_bundle: FeatureBundle
    model_bundle: ModelBundle
