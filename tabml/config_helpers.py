import yaml

from tabml.schemas import feature_config, pipeline_config


def parse_feature_config(yaml_path: str):
    with open(yaml_path) as f:
        config = yaml.load(f)  # config is dict
        return feature_config.FeatureConfig(**config)


def parse_pipeline_config(yaml_path: str):
    with open(yaml_path) as f:
        config = yaml.load(f)  # config is dict
        return pipeline_config.PipelineConfig(**config)
