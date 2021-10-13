"""Utilities that support making inference."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Union

from tabml.config_helpers import parse_pipeline_config
from tabml.experiment_manager import ExperimentManger
from tabml.feature_manager import BaseFeatureManager
from tabml.model_wrappers import initialize_model_wrapper
from tabml.schemas import pipeline_config


@dataclass
class ModelInference:
    feature_manager_cls: BaseFeatureManager
    feature_config_path: str
    transformer_path: str
    model_path: str = ""
    pipeline_config_path: Union[str, None] = None

    def __post_init__(self):
        self._init_feature_manager()
        config = self._get_config()
        self._load_model_wrapper(config.model_wrapper)
        self.features_to_model = list(config.data_loader.features_to_model)

    def _init_feature_manager(self):
        self.fm = self.feature_manager_cls(
            self.feature_config_path, transformer_path=self.transformer_path
        )
        self.fm.load_transformers()

    def _get_config(self):
        pipeline_config_path = (
            self.pipeline_config_path
            or ExperimentManger.get_config_path_from_model_path(self.model_path)
        )
        return parse_pipeline_config(pipeline_config_path)

    def _load_model_wrapper(self, model_wrapper_params: pipeline_config.ModelWrapper):
        self.model_wrapper = initialize_model_wrapper(model_wrapper_params)
        self.model_wrapper.load_model(self.model_path)

    def predict(self, raw_data: List[Dict[str, Any]]) -> Iterable[Any]:
        """Makes prediction for new samples.

        The sample should be given in a form of a list of dictionaries whose keys are
        the raw features. The feature manager transform the raw_data into a processed
        dataframe that is fed into the model to make predictions.
        """
        features = self.fm.transform_new_samples(raw_data, self.features_to_model)
        return self.model_wrapper.predict(features[self.features_to_model])
