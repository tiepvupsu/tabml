from abc import ABC
from pathlib import Path

import mlflow

from tabml import experiment_manager, model_wrappers
from tabml.config_helpers import parse_pipeline_config
from tabml.data_loaders import BaseDataLoader
from tabml.model_analysis import ModelAnalysis
from tabml.utils import factory
from tabml.utils.logger import logger


class BasePipeline(ABC):
    """Base class for pipeline.

    Attributions:
        config:
            A pipeline config object.
        data_loader:
            A data loader object, a subclass of base.data_loaders.BaseDataLoader.
        exp_manager:
            A base.experiment_manager.ExperimentManager object.
    """

    def __init__(self, path_to_config: str, custom_run_dir: str = ""):
        logger.info("=" * 80)
        logger.info(f"Running pipeline with config {path_to_config}")
        logger.info("=" * 80)
        self.exp_manager = experiment_manager.ExperimentManger(
            path_to_config, custom_run_dir=Path(custom_run_dir)
        )
        self.config = parse_pipeline_config(path_to_config)
        self.data_loader = self._get_data_loader()
        assert self.data_loader.label_col is not None, "label_col must be specified"
        self.model_wrapper = model_wrappers.initialize_model_wrapper(
            self.config.model_wrapper
        )

        logger.add(self.exp_manager.get_log_path())

    def run(self):
        self.exp_manager.create_new_run_dir()
        self._init_mlflow()
        with mlflow.start_run():
            self._log_to_mlflow()
            self.train()
            self.analyze_model()

    def _init_mlflow(self):
        model_type = self.model_wrapper.mlflow_model_type
        model_wrappers.MLFLOW_AUTOLOG[model_type]

    def _log_to_mlflow(self):
        model_type = self.model_wrapper.mlflow_model_type
        mlflow.log_param("run_dir", self.exp_manager.run_dir)
        mlflow.log_param("model_type", model_type)
        mlflow.log_params(
            {
                "model_params": self.config.model_wrapper.model_params,
                "fit_params": self.config.model_wrapper.fit_params,
            }
        )

    def train(self):
        model_dir = self.exp_manager.run_dir
        logger.info("Start training the model.")
        self.model_wrapper.fit(self.data_loader, model_dir)

    def _get_data_loader(self) -> BaseDataLoader:
        return factory.create(self.config.data_loader.cls_name)(self.config.data_loader)

    def analyze_model(self) -> None:
        """Analyze the model on the validation dataset.

        The trained model is evaluated based on metrics for predictions slicing by
        each categorical feature specified by features_to_analyze.
        """
        logger.info("Model Analysis")
        assert len(self.config.model_analysis.metrics) > 0, (
            "At least one metrics in model_analysis must be specified. "
            "Add the metrics in model_analysis in the pipeline config"
        )
        assert len(self.config.model_analysis.by_features) > 0, (
            "At least one by_features in model_analysis must be specified. "
            "Add the by_features in model_analysis in the pipeline config"
        )

        ModelAnalysis(
            data_loader=self.data_loader,
            model_wrapper=self.model_wrapper,
            params=self.config.model_analysis,
            output_dir=self.exp_manager.get_model_analysis_dir(),
        ).analyze()
