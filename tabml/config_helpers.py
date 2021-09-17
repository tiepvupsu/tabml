import typing
from pathlib import Path

import yaml
from google.protobuf import json_format, text_format

from tabml.protos import pipeline_pb2
from tabml.schemas import feature_config, pipeline_config


def parse_pipeline_config_pb(pipeline_pbtxt_path: str):
    with open(pipeline_pbtxt_path, "r") as file_object:
        config = pipeline_pb2.Config()
        return text_format.MergeLines(file_object, config)


def pb_to_dict(config):
    return json_format.MessageToDict(config, preserving_proto_field_name=True)


def parse_config_yaml(yaml_path: str):
    with open(yaml_path) as f:
        config = yaml.load(f)  # config is dict
        return feature_config.FeatureConfig(**config)


def parse_feature_config(config_path: str):
    return parse_config_yaml(config_path)


def parse_pipeline_config(yaml_path: typing.Union[str, Path]):
    with open(yaml_path) as f:
        config = yaml.load(f)  # config is dict
        return pipeline_config.PipelineConfig(**config)
