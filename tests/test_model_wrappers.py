from qcore.asserts import assert_eq

from tabml.model_wrappers import get_tabnet_params
from tabml.utils.pb_helpers import parse_pipeline_config_pb
from tabml.utils.utils import write_str_to_file


def test_get_tabnet_params(tmp_path):
    pb_str = """
    config_name: "tabnet"
    data_loader {
        cls_name: "foo"
        features_to_model: "a"
        features_to_model: "b"
        features_to_model: "c"
        features_to_model: "d"
        features_to_model: "e"
    }
    model_wrapper {
        cls_name: "foo_tabnet"
        tabnet_params {
            cat_features {
                feature: "b"
                dim: 3
                emb_dim: 2
            }
            cat_features {
                feature: "e"
                dim: 4
                emb_dim: 1
            }
        }
    }
    trainer {}
    """
    pb_config_path = tmp_path / "tmp.pb"
    write_str_to_file(pb_str, pb_config_path)
    config = parse_pipeline_config_pb(pb_config_path)
    tabnet_params = get_tabnet_params(config)
    expected_params = {"cat_idxs": [1, 4], "cat_dims": [3, 4], "cat_emb_dim": [2, 1]}
    assert_eq(expected_params, tabnet_params)
