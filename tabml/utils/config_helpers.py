import yaml
from attrdict import AttrDict
from google.protobuf import json_format, text_format

from tabml.protos import pipeline_pb2


def parse_pipeline_config_pb(pipeline_pbtxt_path: str):
    with open(pipeline_pbtxt_path, "r") as file_object:
        config = pipeline_pb2.Config()
        return text_format.MergeLines(file_object, config)


def pb_to_dict(config):
    return json_format.MessageToDict(config, preserving_proto_field_name=True)


def parse_config_yaml(yaml_path: str):
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=yaml.BaseLoader)  # config is dict
        return AttrDict(config)


def parse_feature_config(config_path: str):
    ext = config_path.split(".")[-1]
    assert ext in (
        "yaml",
        "yml",
    ), f"feature_path {config_path} extension not supported."
    return parse_config_yaml(config_path)


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
        return parse_config_yaml(pipeline_path)
    else:
        ValueError(f"pipeline_path {pipeline_path} extension not supported.")
