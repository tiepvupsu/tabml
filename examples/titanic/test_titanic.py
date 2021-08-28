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
    em = ExperimentManger(lgbm_config_path)
    last_model_run_dir = em.get_most_recent_run_dir()
    model_path = Path(last_model_run_dir) / "model_0"
    model_inference = ModelInference(
        feature_config_path=feature_config_path,
        pipeline_config_path=lgbm_config_path,
        model_path=model_path,
    )
    raw_data = {
        "PassengerId": 1,
        "Pclass": 1,
        "Name": "First, Dr. Last",
        "Sex": "female",
        "Age": 30,
        "SibSp": 3,
        "Parch": 0,
        "Ticket": 12345,
        "Fare": 10.0,
        "Cabin": None,
        "Embarked": "C",
    }
    model_inference.predict_one_sample(raw_data)
    model_inference.predict_one_batch([raw_data, raw_data])
