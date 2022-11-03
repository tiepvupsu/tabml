from tabml.config_helpers import parse_feature_config
from tabml.examples.housing import feature_manager, pipelines
from tabml.schemas.bundles import FullPipelineBundle
from tabml.utils.utils import change_working_dir_pytest, load_pickle


@change_working_dir_pytest
def test_feature_manager():
    feature_manager.run()


def _get_prediction_feature(prediction_feature, trained_pipeline):
    full_pipeline_bundle: FullPipelineBundle = load_pickle(
        trained_pipeline.exp_manager.get_full_pipeline_path()
    )
    prediction_feature.model_bundle = full_pipeline_bundle.model_bundle
    return prediction_feature


@change_working_dir_pytest
def test_feature_manager_with_prediction_feature():
    feature_config_path = "configs/feature_config.yaml"
    pl_lgbm = pipelines.train_lgbm()
    pl_xgb = pipelines.train_xgboost()
    pl_cat = pipelines.train_catboost()
    feature_config = parse_feature_config(feature_config_path)
    for i, prediction_feature in enumerate(feature_config.prediction_features):
        if prediction_feature.name == "pred_lgbm":
            feature_config.prediction_features[i] = _get_prediction_feature(
                prediction_feature, pl_lgbm
            )
        elif prediction_feature.name == "pred_catboost":
            feature_config.prediction_features[i] = _get_prediction_feature(
                prediction_feature, pl_cat
            )
        elif prediction_feature.name == "pred_xgboost":
            feature_config.prediction_features[i] = _get_prediction_feature(
                prediction_feature, pl_xgb
            )
    fm = feature_manager.FeatureManager(feature_config)
    fm.compute_prediction_features(["pred_lgbm_for_test"])


# @change_working_dir_pytest
# def test_full_pipeline_lgbm():
#     pipelines.train_lgbm()


# @change_working_dir_pytest
# def test_full_pipeline_xgboost():
#     pipelines.train_xgboost()


# @change_working_dir_pytest
# def test_full_pipeline_catboost():
#     pipelines.train_catboost()
