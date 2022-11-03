from tabml.config_helpers import parse_feature_config
from tabml.examples.housing import feature_manager, pipelines
from tabml.schemas.bundles import FullPipelineBundle
from tabml.utils.utils import change_working_dir_pytest, load_pickle


@change_working_dir_pytest
def test_feature_manager():
    feature_manager.run()


@change_working_dir_pytest
def test_feature_manager_with_prediction_feature():
    feature_config_path = "configs/feature_config.yaml"
    pl_lgbm = pipelines.train_lgbm()
    feature_config = parse_feature_config(feature_config_path)
    for i, prediction_feature in enumerate(feature_config.prediction_features):
        if prediction_feature.name == "pred_lgbm":
            full_pipeline_bundle: FullPipelineBundle = load_pickle(
                pl_lgbm.exp_manager.get_full_pipeline_path()
            )
            prediction_feature.model_bundle = full_pipeline_bundle.model_bundle
            feature_config.prediction_features[i] = prediction_feature
    fm = feature_manager.FeatureManager(feature_config)
    fm.compute_prediction_features(["pred_lgbm"])
