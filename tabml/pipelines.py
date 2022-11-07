import pickle
from abc import ABC
from pathlib import Path
from typing import Union

import mlflow

from tabml import experiment_manager, model_wrappers
from tabml.config_helpers import parse_pipeline_config
from tabml.data_loaders import BaseDataLoader
from tabml.feature_manager import BaseFeatureManager
from tabml.model_analysis import ModelAnalysis
from tabml.schemas.pipeline_config import PipelineConfig
from tabml.schemas.bundles import PipelineBundle, ModelBundle
from tabml.utils import factory
from tabml.utils.logger import logger
from tabml.utils.utils import load_pickle, return_or_load


class BasePipeline(ABC):
    """Base class for pipeline.

    Attributions:
        config:
            A pipeline config object or a path to config. If path, then parse the
            pipeline config.
        data_loader:
            A data loader object, a subclass of base.data_loaders.BaseDataLoader.
        exp_manager:
            A base.experiment_manager.ExperimentManager object.
    """

    def __init__(
        self, config: Union[str, Path, PipelineConfig], custom_run_dir: str = ""
    ):
        self.config = return_or_load(config, PipelineConfig, parse_pipeline_config)
        logger.info("=" * 80)
        logger.info(f"Running pipeline with config name {self.config.config_name}")
        logger.info("=" * 80)
        _custom_run_dir = Path(custom_run_dir) if custom_run_dir else None
        self.exp_manager = experiment_manager.ExperimentManager(
            self.config, custom_run_dir=_custom_run_dir
        )
        self.data_loader = self._get_data_loader()
        assert self.data_loader.label_col is not None, "label_col must be specified"
        self.model_wrapper = model_wrappers.initialize_model_wrapper(
            ModelBundle(pipeline_config=self.config, model=None),
        )

        logger.add(self.exp_manager.get_log_path())

    def run(self):
        self.exp_manager.create_new_run_dir()
        self._init_mlflow()
        with mlflow.start_run():
            self._log_to_mlflow()
            self.train()
            self.analyze_model()
        self.save_pipeline_bundle()

    def save_pipeline_bundle(self):
        feature_bundle_path = BaseFeatureManager(
            self.config.data_loader.feature_config_path
        ).get_feature_bundle_path()
        feature_bundle = load_pickle(feature_bundle_path)
        data = PipelineBundle(
            feature_bundle=feature_bundle,
            model_bundle=ModelBundle(
                pipeline_config=self.config,
                model=self.model_wrapper.model,
            ),
        )
        save_path = self.exp_manager.get_pipeline_bundle_path()
        logger.info(f"Saving pipeline bundle to {save_path}")
        with open(save_path, "wb") as pickle_file:
            pickle.dump(data, pickle_file)

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
