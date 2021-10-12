from tabml.utils.utils import change_working_dir_pytest

from . import feature_manager, pipelines


@change_working_dir_pytest
def test_feature_manager():
    feature_manager.run()


@change_working_dir_pytest
def test_full_pipeline_lgbm():
    pipelines.train_lgbm()


@change_working_dir_pytest
def test_full_pipeline_xgboost():
    pipelines.train_xgboost()


@change_working_dir_pytest
def test_full_pipeline_catboost():
    pipelines.train_catboost()
