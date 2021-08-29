"""Utilities that support making inference."""

from typing import Any, Dict, List

from tabml.experiment_manager import ExperimentManger
from tabml.pipelines import BasePipeline


class ModelInference:
    def __init__(self, feature_manager_cls, feature_config_path: str, model_path: str):
        self.fm = feature_manager_cls(feature_config_path)
        pipeline_config_path = ExperimentManger.get_config_path_from_model_path(
            model_path
        )
        self.pipeline = BasePipeline(pipeline_config_path)
        self.pipeline.model_wrapper.load_model(model_path)
        self.features_to_model = list(
            self.pipeline.config.data_loader.features_to_model
        )

    def predict(self, raw_data: List[Dict[str, Any]]) -> List[Any]:
        """Makes prediction for new samples.

        The sample should be given in a form of a list of dictionaries whose keys are
        the raw features. The feature manager transform the raw_data into a processed
        dataframe that is fed into the model to make predictions.
        """
        features = self.fm.transform_new_samples(raw_data, self.features_to_model)
        return self.pipeline.model_wrapper.predict(features[self.features_to_model])
