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
            the pb object of the prototext config file
            (passed via path_to_config when initialized)
        data_loader:
            a data loader object, a subclass of base.data_loaders.BaseDataLoader.
        exp_manager:
            a base.experiment_manager.ExperimentManager object.
        custom_model_wrapper:
            model_wrapper defined by users.
    """

    def __init__(
        self, path_to_config: str, custom_model_wrapper=None, custom_run_dir=""
    ):
        logger.info("=" * 80)
        logger.info(f"Running pipeline with config {path_to_config}")
        logger.info("=" * 80)
        self.exp_manager = experiment_manager.ExperimentManger(
            path_to_config, custom_run_dir=Path(custom_run_dir)
        )
        self.config = parse_pipeline_config(path_to_config)
        self.data_loader = self._get_data_loader()
        assert self.data_loader.label_col is not None, "label_col must be specified"
        self._get_model_wrapper(custom_model_wrapper)

        logger.add(self.exp_manager.get_log_path())

    def run(self):
        self.exp_manager.create_new_run_dir()
        # start mlflow auto log
        model_type = self.model_wrapper.mlflow_model_type
        model_wrappers.MLFLOW_AUTOLOG[model_type]
        with mlflow.start_run():
            mlflow.log_param("model_type", model_type)
            mlflow.log_params(
                {
                    "model_params": self.config.model_wrapper.model_params,
                    "fit_params": self.config.model_wrapper.fit_params,
                }
            )
            self.train()
            self.analyze_model()

    def train(self):
        model_dir = self.exp_manager.run_dir
        logger.info("Start training the model.")
        self.model_wrapper.fit(self.data_loader, model_dir)

    def _get_data_loader(self) -> BaseDataLoader:
        return factory.create(self.config.data_loader.cls_name)(self.config.data_loader)

    def _get_model_wrapper(self, custom_model_wrapper):
        if custom_model_wrapper:
            self.model_wrapper = custom_model_wrapper(self.config.model_wrapper)
        else:
            self.model_wrapper = factory.create(self.config.model_wrapper.cls_name)(
                self.config.model_wrapper
            )

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
