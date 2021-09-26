import typing

import pandas as pd

from tabml.feature_manager import BaseFeatureManager
from tabml.schemas import pipeline_config


class BaseDataLoader:
    """Base class for DataLoader.

    Support extract relevant datasets based on feature manager.
    """

    def __init__(self, params=pipeline_config.DataLoader()):
        self.label_col = params.label_col
        self.features = params.features_to_model
        self.features_and_label = self.features + [self.label_col]
        self.feature_manager = BaseFeatureManager(params.feature_config_path)
        self.feature_manager.load_dataframe()
        self.train_filters = params.train_filters
        self.validation_filters = params.validation_filters

    def _extract_data_and_label(
        self, filters: typing.List[str]
    ) -> typing.Tuple[pd.DataFrame, pd.Series]:
        """Gets data and label."""
        both = self.feature_manager.extract_dataframe(
            features_to_select=self.features_and_label, filters=filters
        )
        label = both.pop(self.label_col)
        return (both, label)

    def get_train_data_and_label(self) -> typing.Tuple[pd.DataFrame, pd.Series]:
        """Gets training data and label."""
        return self._extract_data_and_label(self.train_filters)

    def get_val_data_and_label(self):
        """Gets validation dataset."""
        return self._extract_data_and_label(self.validation_filters)
