from tabml.config_helpers import parse_pipeline_config
from tabml.examples.titanic import feature_manager, pipelines
from tabml.experiment_manager import ExperimentManager
from tabml.inference import ModelInference
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
    pipelines.train_lgbm()
    _test_inference("./configs/lgbm_config.yaml")


@change_working_dir_pytest
def test_full_pipeline_xgboost():
    pipelines.train_xgboost()
    _test_inference("./configs/xgboost_config.yaml")


@change_working_dir_pytest
def test_full_pipeline_catboost():
    pipelines.train_catboost()
    _test_inference("./configs/catboost_config.yaml")


@change_working_dir_pytest
def test_full_pipeline_randomforest():
    config_path = "./configs/rf_config.yaml"
    pipelines.run(config_path)
    _test_inference(config_path)
