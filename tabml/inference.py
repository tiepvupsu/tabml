"""Utilities that support making inference."""

from typing import Any, Dict, List

from tabml.feature_manager import BaseFeatureManager
from tabml.pipelines import BasePipeline


class ModelInference:
    def __init__(
        self,
        feature_config_path: str,
        pipeline_config_path: str,
        model_path: str = None,
    ):
        ...

    def predict_one_sample(self, raw_data: Dict[str, Any]) -> float:
        """
        Makes prediction for one sample.

        The sample should be given in a form of dictionary with keys being the
        raw features.  The raw features should be in the list of base features
        from the feature config.

        The feature manager transform the raw_data into a one-row dataframe that is fed
        into the model to make prediction.

        TODO: Allow raw_data in different forms (json, proto, dataframe, etc)
        """
        ...

    def predict_one_batch(self, raw_batch_data: List[Dict[str, Any]]) -> List[float]:
        ...
