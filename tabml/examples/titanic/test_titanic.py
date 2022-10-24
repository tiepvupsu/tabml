import tempfile
from pathlib import Path
from typing import Union

from tabml.examples.titanic import feature_manager, pipelines
from tabml.experiment_manager import ExperimentManger
from tabml.inference import ModelInference, ModelInferenceCompact
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


def _test_inference(
    config_path,
    test_custom_pipeline_config_path=False,
    transformer_path: Union[Path, str] = "",
):
    feature_config_path = "./configs/feature_config.yaml"
    last_model_run_dir = ExperimentManger(config_path).get_most_recent_run_dir()
    model_path = Path(last_model_run_dir) / "model_0"
    pipeline_config_path = config_path if test_custom_pipeline_config_path else None
    model_inference = ModelInference(
        feature_manager_cls=feature_manager.FeatureManager,
        feature_config_path=feature_config_path,
        transformer_path=transformer_path,
        model_path=model_path,
        pipeline_config_path=pipeline_config_path,
    )
    model_inference.predict(RAW_DATA_SAMPLES)


def _test_inference_compact(config_path):
    full_pipeline_path = (
        ExperimentManger(config_path).get_most_recent_run_dir()
        / ExperimentManger.full_pipeline_filename
    )
    model_inference = ModelInferenceCompact(
        feature_manager_cls=feature_manager.FeatureManager,  # type: ignore
        full_pipeline_path=full_pipeline_path,
    )
    model_inference.predict(RAW_DATA_SAMPLES)


@change_working_dir_pytest
def test_run():
    feature_manager.run()


@change_working_dir_pytest
def test_full_pipeline_lgbm():
    pipelines.train_lgbm()
    _test_inference("./configs/lgbm_config.yaml")
    _test_inference_compact("./configs/lgbm_config.yaml")


@change_working_dir_pytest
def test_full_pipeline_xgboost():
    pipelines.train_xgboost()
    _test_inference("./configs/xgboost_config.yaml")
    _test_inference_compact("./configs/xgboost_config.yaml")


@change_working_dir_pytest
def test_full_pipeline_catboost():
    pipelines.train_catboost()
    _test_inference(
        "./configs/catboost_config.yaml", test_custom_pipeline_config_path=True
    )
    _test_inference_compact("./configs/catboost_config.yaml")


@change_working_dir_pytest
def test_full_pipeline_randomforest():
    config_path = "./configs/rf_config.yaml"
    pipelines.run(config_path)
    _test_inference(config_path, test_custom_pipeline_config_path=True)


@change_working_dir_pytest
def test_custom_transformer_path():
    with tempfile.NamedTemporaryFile() as temp:
        transformer_path = temp.name
        feature_manager.run(transformer_path)
        _test_inference("./configs/lgbm_config.yaml", transformer_path=transformer_path)
