"""Utilities that support making inference."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

from tabml.feature_manager import BaseFeatureManager
from tabml.model_wrappers import initialize_model_wrapper
from tabml.utils.utils import return_or_load, load_pickle
from tabml.schemas.bundles import PipelineBundle


@dataclass
class ModelInference:
    feature_manager_cls: BaseFeatureManager
    pipeline_bundle: Union[str, Path, PipelineBundle]

    def __post_init__(self):
        # Load pickle
        _pipeline_bundle: PipelineBundle = return_or_load(
            self.pipeline_bundle, PipelineBundle, load_pickle
        )
        # Initialize feature_manager
        self.fm = self.feature_manager_cls.from_feature_bundle(
            _pipeline_bundle.feature_bundle
        )
        # Initialize model_wrapper
        pipeline_config = _pipeline_bundle.model_bundle.pipeline_config
        self.model_wrapper = initialize_model_wrapper(_pipeline_bundle.model_bundle)
        self.features_to_model = list(pipeline_config.data_loader.features_to_model)

    def predict(self, raw_data: List[Dict[str, Any]]) -> Iterable[Any]:
        """Makes prediction for new samples.

        The sample should be given in a form of a list of dictionaries whose keys are
        the raw features. The feature manager transform the raw_data into a processed
        dataframe that is fed into the model to make predictions.
        """
        features = self.fm.transform_new_samples(raw_data, self.features_to_model)
        return self.model_wrapper.predict(features[self.features_to_model])
