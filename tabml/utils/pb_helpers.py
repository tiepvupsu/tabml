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
