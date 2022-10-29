from pathlib import Path
from typing import Union

import yaml

from tabml.schemas import feature_config, pipeline_config


def _parse_yaml_as_config(config_cls, yaml_path: Union[str, Path], yaml_str: str = ""):
    if yaml_path and yaml_str:
        raise IOError("Only one of (yaml_path, yam_str) is allowed to be not None.")

    if yaml_path:
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
    else:  # yaml_str is not None
        config = yaml.safe_load(yaml_str)

    return config_cls(**config)


def parse_feature_config(yaml_path: Union[str, Path], yaml_str: str = ""):
    return _parse_yaml_as_config(feature_config.FeatureConfig, yaml_path, yaml_str)


def parse_pipeline_config(yaml_path: Union[str, Path] = "", yaml_str: str = ""):
    return _parse_yaml_as_config(pipeline_config.PipelineConfig, yaml_path, yaml_str)


def save_yaml_config_to_file(
    yaml_config: Union[pipeline_config.PipelineConfig, feature_config.FeatureConfig],
    yaml_path: Union[str, Path],
):
    with open(yaml_path, "w") as yaml_file:
        yaml.dump(yaml_config, yaml_file, default_flow_style=False)
