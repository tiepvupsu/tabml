from tabml.examples.housing import feature_manager, pipelines
from tabml.utils.utils import change_working_dir_pytest


@change_working_dir_pytest
def test_feature_manager():
    feature_manager.run()


@change_working_dir_pytest
def test_feature_manager_with_prediction_feature():
    feature_config_path = "configs/feature_config.yaml"
    fm = feature_manager.FeatureManager.from_config_path(feature_config_path)
    fm.compute_prediction_features(["pred_lgbm_for_test"])


@change_working_dir_pytest
def test_full_pipeline_lgbm():
    pipelines.train_lgbm()


@change_working_dir_pytest
def test_full_pipeline_xgboost():
    pipelines.train_xgboost()


@change_working_dir_pytest
def test_full_pipeline_catboost():
    pipelines.train_catboost()
