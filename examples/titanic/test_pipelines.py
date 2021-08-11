import os

from tabml.utils.utils import change_working_dir_pytest

from . import pipelines


@change_working_dir_pytest
def test_full_pipeline():
    pipelines.train_lgbm()
