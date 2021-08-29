from pathlib import Path

from tabml.experiment_manager import ExperimentManger
from tabml.inference import ModelInference
from tabml.utils.utils import change_working_dir_pytest

from . import feature_manager, pipelines


@change_working_dir_pytest
def test_run():
    feature_manager.run()


@change_working_dir_pytest
def test_full_pipeline():
    pipelines.train_lgbm()


@change_working_dir_pytest
def test_inference():
    feature_config_path = "./configs/feature_config.pb"
    lgbm_config_path = "./configs/lgbm_config.pb"
    last_model_run_dir = ExperimentManger(lgbm_config_path).get_most_recent_run_dir()
    model_path = Path(last_model_run_dir) / "model_0"
    model_inference = ModelInference(
        feature_manager_cls=feature_manager.FeatureManager,
        feature_config_path=feature_config_path,
        model_path=model_path,
    )
    raw_data_samples = [
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
    model_inference.predict(raw_data_samples)
