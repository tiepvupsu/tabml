from google.protobuf import text_format

from tabml.protos import feature_manager_pb2


def parse_feature_config_pb(feature_manager_pbtxt_path: str):
    with open(feature_manager_pbtxt_path, "r") as file_object:
        config = feature_manager_pb2.FeatureManager()
        pb = text_format.MergeLines(file_object, config)
    return pb
