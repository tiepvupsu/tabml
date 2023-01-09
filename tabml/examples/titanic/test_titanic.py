from pathlib import Path
from typing import Union

from tabml.config_helpers import parse_pipeline_config
from tabml.examples.titanic import feature_manager, pipelines
from tabml.experiment_manager import ExperimentManager
from tabml.inference import ModelInference
from tabml.schemas.pipeline_config import PipelineConfig
from tabml.utils.utils import change_working_dir_pytest

RAW_DATA_SAMPLES = [
    {
        "PassengerId": 1,
        "Pclass": 1,
        "Name": "First, Mr. Last",
        "Sex": "female",
        "SibSp": 3,
        "Parch": 0,
        "Ticket": 12345,
        "Fare": 10.0,
        "Cabin": None,
        "Embarked": "C",
    },
    {
        "PassengerId": 2,
        "Pclass": 1,
        "Name": "First, Mrs. Last",
        "Sex": "male",
        "Age": 60,
        "SibSp": 0,
        "Parch": 2,
        "Ticket": 12345,
        "Fare": 100.0,
        "Cabin": None,
        "Embarked": "Q",
    },
]


def create_test_config(config_path: Union[str, Path]) -> PipelineConfig:
    config = parse_pipeline_config(config_path)
    config.model_wrapper.model_params["n_estimators"] = 10
    config.model_analysis.explainer_train_size = 50
    config.model_analysis.metric_train_size = 10
    return config


def _test_inference(config_path):
    config = parse_pipeline_config(yaml_path=config_path)
    pipeline_bundle_path = (
        ExperimentManager(config).get_most_recent_run_dir()
        / ExperimentManager.pipeline_bundle_filename
    )
    model_inference = ModelInference(
        feature_manager_cls=feature_manager.FeatureManager,  # type: ignore
        pipeline_bundle=pipeline_bundle_path,
    )
    model_inference.predict(RAW_DATA_SAMPLES)


@change_working_dir_pytest
def test_run():
    feature_manager.run()


@change_working_dir_pytest
def test_full_pipeline_lgbm():
    pipeline_config_path = "configs/lgbm_config.yaml"
    config = create_test_config(pipeline_config_path)
    pipelines.run(config)
    _test_inference(pipeline_config_path)


@change_working_dir_pytest
def test_full_pipeline_xgboost():
    pipeline_config_path = "./configs/xgboost_config.yaml"
    config = create_test_config(pipeline_config_path)
    config.model_analysis.show_feature_importance = False
    pipelines.run(config)
    _test_inference(pipeline_config_path)


@change_working_dir_pytest
def test_full_pipeline_catboost():
    pipeline_config_path = "./configs/catboost_config.yaml"
    config = create_test_config(pipeline_config_path)
    config.model_analysis.show_feature_importance = False
    pipelines.run(config)
    _test_inference(pipeline_config_path)


@change_working_dir_pytest
def test_full_pipeline_randomforest():
    pipeline_config_path = "./configs/rf_config.yaml"
    config = create_test_config(pipeline_config_path)
    config.model_analysis.show_feature_importance = False
    pipelines.run(config)
    _test_inference(pipeline_config_path)


@change_working_dir_pytest
def test_fill_pipeline_exp_weight_lgbm():
    pipeline_config_path = "configs/lgbm_config.yaml"
    config = create_test_config(pipeline_config_path)
    config.model_analysis.show_feature_importance = False
    config.data_loader.feature_to_create_weights = "imputed_age"
    config.model_wrapper.cls_name = (
        "tabml.model_wrappers.ExponentialWeightLgbmClassifierModelWrapper"
    )
    config.model_wrapper.weight_params = {
        "scale": 1,
        "decay": 20,
        "num_same_weight_samples": 1,
    }
    pipelines.run(config)
    _test_inference(pipeline_config_path)
