from qcore.asserts import assert_eq

from tabml.config_helpers import parse_pipeline_config
from tabml.model_wrappers import get_tabnet_params
from tabml.utils.utils import write_str_to_file


def test_get_tabnet_params(tmp_path):
    config_str = """
    config_name: "tabnet"
    data_loader:
        name: "foo"
        feature_manager_config_path: "bar"
        features_to_model: ["a", "b", "c", "d", "e"]
        label_col: "f"
        train_filters: ["g"]
        validation_filters: ["h"]
    model_wrapper:
        name: "foo_tabnet"
        model_params:
            cat_features:
                - feature: "b"
                  dim: 3
                  emb_dim: 2
                - feature: "e"
                  dim: 4
                  emb_dim: 1
    model_analysis:
        metrics: ["foo"]
        by_features: ["a"]
    """
    config_path = tmp_path / "tmp.yaml"
    write_str_to_file(config_str, config_path)
    config = parse_pipeline_config(config_path)
    tabnet_params = get_tabnet_params(config)
    expected_params = {"cat_idxs": [1, 4], "cat_dims": [3, 4], "cat_emb_dim": [2, 1]}
    assert_eq(expected_params, tabnet_params)
