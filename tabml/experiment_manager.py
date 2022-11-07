import datetime
import re
from pathlib import Path
from typing import Union

from tabml.config_helpers import parse_pipeline_config, save_yaml_config_to_file
from tabml.schemas.pipeline_config import PipelineConfig
from tabml.utils.utils import return_or_load


class ExperimentManager:
    """Class managing folder structure of an experiment.

    For each experiment, there will be one run_dir under exp_root_dir that contains
    all related information about the run, e.g. run log, config file, submission
    csv file, trained model, and a model_analysis folder.

    Attributes:
        exp_root_dir: root directory of all experiments.
        run_prefix: prefix of the run name subfolder inside exp_root_dir.
        run_dir: run dir name (run_prefix + timestamp).
    """

    log_filename = "run.log"
    config_filename = "config.yaml"
    _model_analysis_dir = "model_analysis"
    pipeline_bundle_filename = "full_pipeline.pickle"

    def __init__(
        self,
        config: Union[str, Path, PipelineConfig],
        should_create_new_run_dir: bool = True,
        exp_root_dir: Path = Path("./experiments"),
        custom_run_dir: Union[None, Path] = None,
    ):
        """
        Args:
            config: PipelineConfig object or path to the yaml configuration.
            should_create_new_run_dir: create new experiment subfolder (True) or not
                (False). If not, set the experiment subfolder to the most recent run.
            run_prefix: prefix of the run name subfolder inside exp_root_dir
            run_dir: run dir name (exp_root_dir/run_prefix + timestamp)
            custom_run_dir: custom run dir that user can specify
        """
        self.config = return_or_load(config, PipelineConfig, parse_pipeline_config)

        self.exp_root_dir = exp_root_dir
        self.run_prefix = self.config.config_name + "_"
        self.custom_run_dir = custom_run_dir
        if not custom_run_dir or not custom_run_dir.name:
            self.run_dir = self._get_run_dir(should_create_new_run_dir)
        else:
            self.run_dir = custom_run_dir

    def _get_run_dir(self, should_create_new_run_dir):
        if not should_create_new_run_dir:
            return self.get_most_recent_run_dir()
        return self.exp_root_dir.joinpath(self.run_prefix + _get_time_stamp())

    def create_new_run_dir(self):
        _make_dir_if_needed(self.run_dir)
        self._save_config_to_file()

    def _save_config_to_file(self):
        save_yaml_config_to_file(self.config, self.get_config_path())

    def get_log_path(self):
        return self._make_path_under_run_dir(self.log_filename)

    def get_config_path(self):
        return self._make_path_under_run_dir(self.config_filename)

    def get_pipeline_bundle_path(self):
        return self._make_path_under_run_dir(self.pipeline_bundle_filename)

    def get_model_analysis_dir(self):
        res = self._make_path_under_run_dir(self._model_analysis_dir)
        _make_dir_if_needed(res)
        return res

    def _make_path_under_run_dir(self, sub_path: str) -> Path:
        return self.run_dir.joinpath(sub_path)

    def get_most_recent_run_dir(self):
        """Returns the run_dir corresponding to the most recent timestamp.

        Raises:
            IOError if there is no such folder
        """
        if self.custom_run_dir:
            ValueError("get_most_recent_run_dir does not support custom run dir")
        subfolders = sorted(
            [
                sub
                for sub in self.exp_root_dir.iterdir()
                if sub.is_dir()
                and sub.name.startswith(self.run_prefix)
                and bool(re.match("[0-9]{6}_[0-9]{6}", sub.name[-13:]))
            ],
            key=lambda x: x.name[-13:],  # YYmmDD_HHMMSS
        )
        if not subfolders:
            raise IOError(
                "Could not find any run directory starting with "
                f"{self.exp_root_dir.joinpath(self.run_prefix)}"
            )
        return subfolders[-1]

    @classmethod
    def get_config_path_from_model_path(cls, model_path: str) -> str:
        run_dir = Path(model_path).parents[0]
        return str(run_dir / cls.config_filename)


def _make_dir_if_needed(dir_path: Union[str, Path]):
    if not Path(dir_path).exists():
        Path(dir_path).mkdir(parents=True)


def _get_time_stamp() -> str:
    """Returns a time stamp string in format 'YYmmdd_HHMMSS'.

    Example: 200907_123344.
    """
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")[2:]
