from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from tabml.feature_config_helper import FeatureConfigHelper
from tabml.utils.logger import logger
from tabml.utils.utils import check_uniqueness


class BaseFeatureManager:
    """A Base class for feature manager.

    Attributes:
        config_helper: A FeatureConfigHelper instance to manage feature dependency.
        raw_data_dir: A directory of raw data (set in feature manager config).
        dataset_name: Name of the final dataset (set in feature manager config).
        dataset_path: A string of the full path to where the dataframe is saved.
        is_pandas: A boolen value indicating the dataframe is pandas or not (spark).
        features_in_config: List of features specified in feature manager config.
        raw_data:
            A dictionary with values being supporting dataframes for each dataset.
            Should be defined in method load_raw_data().
        dataframe: The main dataframe.
        feature_metadata:
            A dict of {feature_name: (index, dtype, list of its dependents)}.
        base_transforming_feature_class:
            A direct subclass of BaseTransformingFeature, is defined in each project.
        transforming_class_by_feature_name:
            A dictionary of
            {feature_name: a transforming feature generating that feature}.
            All the transforming features must be direct subclasses of
            self.base_transforming_feature_class.
    """

    def __init__(self, pb_config_path: str, is_pandas: bool = True):
        self.config_helper = FeatureConfigHelper(pb_config_path)
        self.raw_data_dir = self.config_helper.raw_data_dir
        self.dataset_name = self.config_helper.dataset_name
        self.feature_metadata = self.config_helper.feature_metadata
        self.dataset_path = (
            Path(self.raw_data_dir) / "features" / f"{self.dataset_name}.csv"
        )
        self.is_pandas = is_pandas
        self.features_in_config: List[str] = self.config_helper.all_features
        self.raw_data: Dict[str, Any] = {}
        self.dataframe: pd.DataFrame = pd.DataFrame()
        self.base_transforming_feature_class = self._get_base_transforming_class()
        self.transforming_class_by_feature_name = self._get_transforming_class_by_name()

    def _get_base_transforming_class(self):
        # should be implemented in subclasses.
        raise NotImplementedError

    def _get_transforming_class_by_name(self) -> Dict[str, Any]:
        if self.base_transforming_feature_class is None:
            raise NotImplementedError
        transforming_classes = self.base_transforming_feature_class.__subclasses__()
        check_uniqueness(
            [transforming_class.name for transforming_class in transforming_classes]
        )
        return {
            transforming_class.name: transforming_class
            for transforming_class in transforming_classes
        }


class BaseTransformingFeature(ABC):
    """Base class for transforming features.
    In each project, users need to create a subclass of BaseTransformingFeature. All
    transforming features in one project must be subclasses of that subclass.
    Attributes:
        name:
            (class attribute) This is the feature name that the class computes. Must be
            unique within a project.
    """

    name = ""

    def __init__(self, dependencies: List[str], raw_data: Dict):
        self.dependencies = dependencies
        self.raw_data = raw_data

    def _transform(self, dataframe):
        logger.info(f"Computing feature {self.name} in pandas ...")
        # This is to make sure that _transform_ methods do not use any columns other
        # than dependencies.
        df = dataframe[self.dependencies]
        return self._transform_pandas(df)

    @abstractmethod
    def transform(self, df):
        raise NotImplementedError("Must be implemented in subclasses.")
