from tabml.utils.utils import change_working_dir_pytest

from . import feature_manager, pipelines


@change_working_dir_pytest
def test_full_pipeline():
    feature_manager.run()
    pipelines.train_lgbm()
