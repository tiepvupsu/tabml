import datetime
import os
import re
import shutil
from pathlib import Path

from tabml.utils.pb_helpers import parse_pipeline_config_pb


class ExperimentManger:
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
    config_filename = "config.pb"
    _model_analysis_dir = "model_analysis"

    def __init__(
        self,
        path_to_config: str,
        should_create_new_run_dir: bool = True,
        exp_root_dir: str = "./experiments",
    ):
        """
        Args:
            path_to_config: path to proto pipeline config file.
            should_create_new_run_dir: create new experiment subfolder (True) or not
                (False). If not, set the experiment subfolder to the most recent run.
            run_prefix: prefix of the run name subfolder inside exp_root_dir
            run_dir: run dir name (run_prefix + timestamp)
        """
        self._path_to_config = path_to_config
        self._config = parse_pipeline_config_pb(self._path_to_config)
        self.exp_root_dir = exp_root_dir
        self.run_prefix = self._config.config_name + "_"
        self.run_dir = self._get_run_dir(should_create_new_run_dir)

    def _get_run_dir(self, should_create_new_run_dir):
        if not should_create_new_run_dir:
            return self.get_most_recent_run_dir()
        return os.path.join(self.exp_root_dir, self.run_prefix + _get_time_stamp())

    def create_new_run_dir(self):
        _make_dir_if_needed(self.run_dir)
        self._copy_config_file()

    def _copy_config_file(self):
        shutil.copyfile(self._path_to_config, self.get_config_path())

    def get_log_path(self):
        return self._make_path(self.log_filename)

    def get_config_path(self):
        return self._make_path(self.config_filename)

    def get_model_analysis_dir(self):
        res = self._make_path(self._model_analysis_dir)
        _make_dir_if_needed(res)
        return res

    def _make_path(self, filename: str) -> str:
        return os.path.join(self.run_dir, filename)

    def get_most_recent_run_dir(self):
        """Return the run_dir corresponding to the most recent timestamp.

        Raises:
            IOError if there is no such folder
        """
        subfolders = sorted(
            [
                sub
                for sub in os.listdir(self.exp_root_dir)
                # if os.path.isdir(os.path.join(self.exp_root_dir, sub))
                if (Path(self.exp_root_dir) / sub).is_dir()
                and sub.startswith(self.run_prefix)
                and bool(re.match("[0-9]{6}_[0-9]{6}", sub[-13:]))
            ],
            key=lambda x: x[-13:],  # YYmmDD_HHMMSS
        )
        if not subfolders:
            raise IOError(
                "Could not find any run directory starting with "
                f"{Path(self.exp_root_dir) / self.run_prefix}"
            )
        sub = subfolders[-1]
        most_recent_run_dir = os.path.join(self.exp_root_dir, sub)
        return most_recent_run_dir


def _make_dir_if_needed(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def _get_time_stamp() -> str:
    """Returns a time stamp string in format 'YYmmdd_HHMMSS'.

    Example: 200907_123344.
    """
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")[2:]
