"""Utilities that support making inference."""

from typing import Any, Dict, List

from tabml.feature_manager import BaseFeatureManager
from tabml.pipelines import BasePipeline


class ModelInference:
    def __init__(
        self,
        feature_config_path: str,
        feature_manager_cls: BaseFeatureManager,
        pipeline_config_path: str,
        model_path: str = None,
    ):
        self.fm = feature_manager_cls(feature_config_path)
        self.pipeline = BasePipeline(pipeline_config_path)
        self.pipeline.model_wrapper.load_model(model_path)

    @classmethod
    def init_from_model_path(cls, model_path):
        """
        feature_config_path and pipelien_config_path could be inferred from model_path
        """
        # TODO: implement me
        NotImplementedError

    def predict(self, raw_data: List[Dict[str, Any]]) -> List[Any]:
        """
        Makes prediction for new samples.

        The sample should be given in a form of a list of dictionaries whose keys are
        the raw features. The feature manager transform the raw_data into a processed dataframe that
        is fed into the model to make predictions.
        """
        features_to_model = list(self.pipeline.config.data_loader.features_to_model)
        self.fm.get_raw_data_one_sample(raw_data)
        features = self.fm.transform_new_samples(raw_data, features_to_model)
        return self.pipeline.model_wrapper.predict(features[features_to_model])
