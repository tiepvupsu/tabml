from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from tabml.feature_config_helper import FeatureConfigHelper
from tabml.protos import feature_manager_pb2
from tabml.utils.logger import logger
from tabml.utils.utils import check_uniqueness

PANDAS_DTYPE_MAPPING = {
    feature_manager_pb2.BOOL: "bool",
    feature_manager_pb2.INT32: "int32",
    feature_manager_pb2.INT64: "int64",
    feature_manager_pb2.STRING: "str",
    feature_manager_pb2.FLOAT: "float32",
    feature_manager_pb2.DOUBLE: "float64",
    # DATETIME will be converted to datetime parse_date https://tinyurl.com/y4waw6np
    feature_manager_pb2.DATETIME: "datetime64[ns]",
}


class BaseFeatureManager(ABC):
    """A Base class for feature manager.

    Attributes:
        config_helper: A FeatureConfigHelper instance to manage feature dependency.
        raw_data_dir: A directory of raw data (set in feature manager config).
        dataset_name: Name of the final dataset (set in feature manager config).
        dataset_path: A string of the full path to where the dataframe is saved.
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

    def __init__(self, pb_config_path: str):
        self.config_helper = FeatureConfigHelper(pb_config_path)
        self.raw_data_dir = self.config_helper.raw_data_dir
        self.dataset_name = self.config_helper.dataset_name
        self.feature_metadata = self.config_helper.feature_metadata
        self.dataset_path = (
            Path(self.raw_data_dir) / "features" / f"{self.dataset_name}.csv"
        )
        self.features_in_config: List[str] = self.config_helper.all_features
        self.raw_data: Dict[str, Any] = {}
        self.dataframe: pd.DataFrame = pd.DataFrame()
        self.base_transforming_feature_class = self._get_base_transforming_class()
        self.transforming_class_by_feature_name = self._get_transforming_class_by_name()

    @abstractmethod
    def _get_base_transforming_class(self):
        # should be implemented in subclasses.
        pass

    def _get_transforming_class_by_name(self) -> Dict[str, Any]:
        transforming_classes = self.base_transforming_feature_class.__subclasses__()
        check_uniqueness(
            [transforming_class.name for transforming_class in transforming_classes]
        )
        return {
            transforming_class.name: transforming_class
            for transforming_class in transforming_classes
        }

    def compute_feature(self, feature_name: str) -> None:
        """Computes one feature column.

        This method should be used only when a new feature is added to the dataframe.
        If the feature exists, method update_feature should be called to make sure that
        all dependents of feature_name are also udpated.
        """
        if self.dataframe is None:
            raise NotImplementedError(
                "self.dataframe must be initialized in self.initialize_dataframe()"
            )
        assert feature_name not in self.dataframe.columns, (
            f"Feature {feature_name} already exists in the dataframe. Do you want to "
            "update it (using update_feature() method) instead?"
        )
        self._compute_feature(feature_name)

    def _compute_feature(self, feature_name: str) -> None:
        if self.dataframe is None:
            raise NotImplementedError(
                "self.dataframe must be initialized in self.initialize_dataframe()"
            )
        transforming_class = self.transforming_class_by_feature_name[feature_name]
        series = transforming_class(
            dependencies=self.config_helper.find_dependencies(feature_name),
            raw_data=self.raw_data,
        ).transform(self.dataframe)
        dtype = self.feature_metadata[feature_name].dtype
        if dtype == feature_manager_pb2.DATETIME:
            self.dataframe[feature_name] = pd.to_datetime(series)
        else:
            self.dataframe[feature_name] = pd.Series(
                series, dtype=PANDAS_DTYPE_MAPPING[dtype]
            )

    def update_feature(self, feature_name: str):
        """Updates one feature in the dataframe.

        If this is an existing feature, all of its dependents should be computed.
        """
        features_to_compute = [feature_name]
        features_to_compute.extend(self.config_helper.find_dependents(feature_name))
        for feature_name in features_to_compute:
            self._compute_feature(feature_name)


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
        return self.transform(df)

    @abstractmethod
    def transform(self, df):
        pass
