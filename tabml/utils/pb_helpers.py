import pathlib

from google.protobuf import json_format, text_format

from tabml.protos import feature_manager_pb2, path_pb2, pipeline_pb2
from tabml.utils import utils


def parse_feature_config_pb(feature_manager_pbtxt_path: str):
    with open(feature_manager_pbtxt_path, "r") as file_object:
        config = feature_manager_pb2.FeatureConfig()
        return text_format.MergeLines(file_object, config)


def parse_pipeline_config_pb(pipeline_pbtxt_path: str):
    with open(pipeline_pbtxt_path, "r") as file_object:
        config = pipeline_pb2.Config()
        return text_format.MergeLines(file_object, config)


def get_absolute_path(path: path_pb2.Path) -> pathlib.Path:
    """Converts a path_pb2.Path path to the fully absolute path."""
    if path.is_absolute_path:
        return path.path
    return utils.get_full_path(path.path)


def pb_to_dict(config):
    return json_format.MessageToDict(config, preserving_proto_field_name=True)
