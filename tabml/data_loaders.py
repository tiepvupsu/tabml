import typing

import pandas as pd

from tabml.feature_manager import BaseFeatureManager


class BaseDataLoader:
    """Base class for DataLoader.

    Support extract relevant dataset based on feature manager.
    """

    def __init__(self, config):
        self.config = config
        self.label_col = self.config.data_loader.label_col
        self.features = list(self.config.data_loader.features_to_model)
        self.features_and_label = self.features + [self.label_col]
        self.feature_manager = self._get_feature_manager()
        self.feature_manager.load_dataframe()

    def _get_feature_manager(self):
        fm_config_path = self.config.data_loader.feature_manager_config_path
        return BaseFeatureManager(fm_config_path)

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
        return self._extract_data_and_label(self.config.data_loader.train_filters)

    def get_val_data_and_label(self):
        """Gets validation dataset."""
        return self._extract_data_and_label(self.config.data_loader.validation_filters)
