import yaml
from attrdict import AttrDict
from google.protobuf import json_format, text_format

from tabml.protos import feature_manager_pb2, pipeline_pb2


def parse_feature_config_pb(feature_manager_pbtxt_path: str):
    with open(feature_manager_pbtxt_path, "r") as file_object:
        config = feature_manager_pb2.FeatureConfig()
        return text_format.MergeLines(file_object, config)


def parse_pipeline_config_pb(pipeline_pbtxt_path: str):
    with open(pipeline_pbtxt_path, "r") as file_object:
        config = pipeline_pb2.Config()
        return text_format.MergeLines(file_object, config)


def pb_to_dict(config):
    return json_format.MessageToDict(config, preserving_proto_field_name=True)


def parse_pipeline_config_yaml(pipeline_yaml_path: str):
    with open(pipeline_yaml_path) as f:
        config = yaml.load(f, Loader=yaml.BaseLoader)  # config is dict
        return AttrDict(config)


def parse_pipeline_config(pipeline_path: str):
    ext = pipeline_path.split(".")[-1]
    assert ext in (
        "pb",
        "yaml",
        "yml",
    ), f"pipeline_path {pipeline_path} extension not supported."
    if ext == "pb":
        return parse_pipeline_config_pb(pipeline_path)
    elif ext in ("yaml", "yml"):
        return parse_pipeline_config_yaml(pipeline_path)
    else:
        ValueError(f"pipeline_path {pipeline_path} extension not supported.")


class PipelineConfig:
    def __init__(self, config_path: str):
        ext = config_path.split(".")[-1]
        if ext == "pb":
            self.type = "proto"
        elif ext in ("yaml", "yml"):
            self.type == "yaml"
        else:
            ValueError(f"config type ({config_path}) is not supported.")
        self.config_path = config_path
