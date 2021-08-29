from tabml.utils.utils import change_working_dir_pytest

from . import feature_manager, pipelines


@change_working_dir_pytest
def test_run():
    feature_manager.run()


@change_working_dir_pytest
def test_full_pipeline():
    pipelines.train_lgbm()


@change_working_dir_pytest
def test_fit_example():
    pb_config_path = "configs/feature_config.pb"
    fm = feature_manager.FeatureManager(pb_config_path)
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
    transforming_features = [
        "imputed_age",
        "bucketized_age",
        "min_max_scaled_age",
        "title",
        "coded_title",
        "coded_sex",
    ]
    print(
        fm.transform_new_samples(
            raw_data_samples, transforming_features=transforming_features
        )
    )
