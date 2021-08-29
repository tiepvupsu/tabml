from pathlib import Path

from tabml.experiment_manager import ExperimentManger
from tabml.inference import ModelInference
from tabml.pipelines import BasePipeline


def train_lgbm():
    path_to_config = "configs/lgbm_config.pb"
    pipeline = BasePipeline(path_to_config)
    pipeline.run()
    pipeline.analyze_model()


def inference():
    feature_config_path = "./configs/feature_config.pb"
    lgbm_config_path = "./configs/lgbm_config.pb"
    last_model_run_dir = ExperimentManger(lgbm_config_path).get_most_recent_run_dir()
    model_path = Path(last_model_run_dir) / "model_0"
    model_inference = ModelInference(
        feature_config_path=feature_config_path,
        feature_manager_cls=feature_manager.FeatureManager,
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
    model_inference.predict_one_sample([raw_data, raw_data])


if __name__ == "__main__":
    # train_lgbm()
    inference()
