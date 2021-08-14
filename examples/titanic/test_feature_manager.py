from tabml.utils.utils import change_working_dir_pytest

from . import feature_manager


@change_working_dir_pytest
def test_run():
    feature_manager.run()
