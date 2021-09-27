"""Utilities that support making inference."""

from typing import Any, Dict, List, Union

from tabml.config_helpers import parse_pipeline_config
from tabml.experiment_manager import ExperimentManger
from tabml.utils import factory


class ModelInference:
    def __init__(
        self,
        feature_manager_cls,
        feature_config_path: str,
        model_path: str,
        pipeline_config_path: Union[str, None] = None,
        custom_model_wrapper=None,
        transformer_path=None,
    ):
        self.fm = feature_manager_cls(
            feature_config_path, transformer_path=transformer_path
        )
        self.fm.load_transformers()
        if pipeline_config_path is None:
            pipeline_config_path = ExperimentManger.get_config_path_from_model_path(
                model_path
            )
        self.config = parse_pipeline_config(pipeline_config_path)
        if custom_model_wrapper:
            self.model_wrapper = custom_model_wrapper(self.config.model_wrapper)
        else:
            self.model_wrapper = factory.create(self.config.model_wrapper.cls_name)(
                self.config.model_wrapper
            )
        self.model_wrapper.load_model(model_path)
        self.features_to_model = list(self.config.data_loader.features_to_model)

    def predict(self, raw_data: List[Dict[str, Any]]) -> List[Any]:
        """Makes prediction for new samples.

        The sample should be given in a form of a list of dictionaries whose keys are
        the raw features. The feature manager transform the raw_data into a processed
        dataframe that is fed into the model to make predictions.
        """
        features = self.fm.transform_new_samples(raw_data, self.features_to_model)
        return self.model_wrapper.predict(features[self.features_to_model])
