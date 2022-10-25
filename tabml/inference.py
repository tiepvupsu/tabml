"""Utilities that support making inference."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

from tabml.config_helpers import parse_pipeline_config
from tabml.experiment_manager import ExperimentManger
from tabml.feature_manager import BaseFeatureManager
from tabml.model_wrappers import (
    initialize_model_wrapper,
    initialize_model_wrapper_from_full_pipeline_pickle,
)
from tabml.utils.utils import load_pickle


def _predict(fm, model_wrapper, features_to_model, raw_data):
    """Makes prediction for new samples.

    The sample should be given in a form of a list of dictionaries whose keys are
    the raw features. The feature manager transform the raw_data into a processed
    dataframe that is fed into the model to make predictions.
    """
    features = fm.transform_new_samples(raw_data, features_to_model)
    return model_wrapper.predict(features[features_to_model])


@dataclass
class ModelInference:
    feature_manager_cls: BaseFeatureManager
    full_pipeline_path: Union[str, Path]

    def __post_init__(self):
        # Load pickle
        data = load_pickle(self.full_pipeline_path)
        # Initialize feature_manager
        self.fm = self.feature_manager_cls.from_full_pipeline_data(data)
        # Initialize model_wrapper
        pipeline_config = data["pipeline_config"]
        self.model_wrapper = initialize_model_wrapper_from_full_pipeline_pickle(data)
        self.features_to_model = list(pipeline_config.data_loader.features_to_model)

    def predict(self, raw_data: List[Dict[str, Any]]) -> Iterable[Any]:
        return _predict(self.fm, self.model_wrapper, self.features_to_model, raw_data)


def _get_config(pipeline_config_path, model_path):
    pipeline_config_path = (
        pipeline_config_path
        or ExperimentManger.get_config_path_from_model_path(model_path)
    )
    return parse_pipeline_config(pipeline_config_path)
