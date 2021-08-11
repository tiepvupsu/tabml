from abc import ABC

from tabml import experiment_manager
from tabml.data_loaders import BaseDataLoader
from tabml.model_analysis import ModelAnalysis
from tabml.trainers import BaseTrainer
from tabml.utils import factory
from tabml.utils.logger import logger
from tabml.utils.pb_helpers import parse_pipeline_config_pb


class BasePipeline(ABC):
    """Base class for pipeline.

    Attributions:
        config:
            the pb object of the prototext config file
            (passed via path_to_config when initialized)
        data_loader:
            a data loader object, a subclass of base.data_loaders.BaseDataLoader.
        trainer:
            a trainer object, a subclass of base.trainers.BaseTrainer.
        exp_manager:
            a base.experiment_manager.ExperimentManager object.
    """

    def __init__(self, path_to_config: str):
        logger.info("=" * 80)
        logger.info(f"Running pipeline with config {path_to_config}")
        logger.info("=" * 80)
        self.exp_manager = experiment_manager.ExperimentManger(path_to_config)
        self.config = parse_pipeline_config_pb(path_to_config)
        self.data_loader = self._get_data_loader()
        assert self.data_loader.label_col is not None, "label_col must be specified"
        self._get_model_wrapper()
        self.trainer = self._get_trainer()

        logger.add(self.exp_manager.get_log_path())

    def run(self):
        self.exp_manager.create_new_run_dir()
        self.train()

    def train(self):
        model_dir = self.exp_manager.run_dir
        logger.info("Start training the model.")
        self.trainer.train(model_dir)

    def _get_data_loader(self) -> BaseDataLoader:
        return factory.create(self.config.data_loader.cls_name)(self.config)

    def _get_model_wrapper(self):
        self.model_wrapper = factory.create(self.config.model_wrapper.cls_name)(
            self.config
        )

    def _get_trainer(self) -> BaseTrainer:
        return factory.create(self.config.trainer.cls_name)(
            self.model_wrapper, self.data_loader, self.config
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
            self.data_loader,
            self.model_wrapper,
            self.config.model_analysis.by_features,
            metric_names=self.config.model_analysis.metrics,
            output_dir=self.exp_manager.get_model_analysis_dir(),
        ).analyze()
