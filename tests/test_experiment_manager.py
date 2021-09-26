from pathlib import Path

import pytest
from qcore.asserts import AssertRaises, assert_eq

from tabml import experiment_manager
from tabml.utils.utils import write_str_to_file


class TestExperimentManager:
    @pytest.fixture(autouse=True)
    def setup_class(self, tmp_path):
        pb_str = """
            config_name: "foo"
            data_loader:
              cls_name: "bar"
              feature_config_path: "rab"
              features_to_model: ["c"]
              label_col: "d"
              train_filters: []
              validation_filters: []
            model_wrapper:
              cls_name: "bobar"
            model_analysis:
              metrics: ["a"]
              by_features: ["b"]
        """
        self.pipeline_config_path = tmp_path / "pipeline_config.pb"
        write_str_to_file(pb_str, self.pipeline_config_path)

    def test_not_create_new_run_dir(self, tmp_path):
        exp_root_dir = Path(tmp_path) / "exp"
        exp_manager = experiment_manager.ExperimentManger(
            self.pipeline_config_path, exp_root_dir=exp_root_dir
        )
        exp_manager.create_new_run_dir()
        experiment_manager.ExperimentManger(
            self.pipeline_config_path,
            should_create_new_run_dir=False,
            exp_root_dir=exp_root_dir,
        )

    def test_not_creat_run_dir_not_exist(self, tmp_path):
        exp_root_dir = Path(tmp_path) / "exp2"
        Path(exp_root_dir).mkdir(parents=True)

        with AssertRaises(IOError) as assert_raises:
            experiment_manager.ExperimentManger(
                self.pipeline_config_path,
                should_create_new_run_dir=False,
                exp_root_dir=exp_root_dir,
            )

        error_message = assert_raises.expected_exception_found
        assert_eq(
            True,
            error_message.args[0].startswith(
                "Could not find any run directory starting with"
            ),
        )
