from pathlib import Path
from typing import Union

import yaml

from tabml.schemas import feature_config, pipeline_config


def parse_feature_config(yaml_path: Union[str, Path]):
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
        return feature_config.FeatureConfig(**config)


def parse_pipeline_config(yaml_path: Union[str, Path]):
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
        return pipeline_config.PipelineConfig(**config)
