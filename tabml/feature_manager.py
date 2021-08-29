import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
        transformer_dict:
            A dictionary of {feature_name: its transformer}. This transformer_dict is
            formed and saved to a pickle in the "training" stage. In "serving" stage,
            it's loaded and does the transformations.
        transformer_path:
            pickle path to save transformers.
    """

    def __init__(self, pb_config_path: str):
        self.config_helper = FeatureConfigHelper(pb_config_path)
        self.raw_data_dir = self.config_helper.raw_data_dir
        self.dataset_name = self.config_helper.dataset_name
        self.feature_metadata = self.config_helper.feature_metadata
        self.dataset_path = self.get_dataset_path()
        self.features_in_config: List[str] = self.config_helper.all_features
        self.raw_data: Dict[str, Any] = {}
        self.dataframe: pd.DataFrame = pd.DataFrame()
        self.transforming_class_by_feature_name: Dict[str, Any] = {}
        self.transformer_dict: Dict[str, Any] = {}
        self.transformer_path = self.get_transformer_path()

    def get_dataset_path(self):
        return Path(self.raw_data_dir) / "features" / f"{self.dataset_name}.csv"

    def get_transformer_path(self):
        return Path(self.raw_data_dir) / "features" / "transformers.pickle"

    def _get_base_transforming_class(self):
        raise NotImplementedError

    def _get_transforming_class_by_name(self):
        base_transforming_feature_class = self._get_base_transforming_class()
        transforming_classes = base_transforming_feature_class.__subclasses__()
        check_uniqueness(
            [transforming_class.name for transforming_class in transforming_classes]
        )
        self.transforming_class_by_feature_name = {
            transforming_class.name: transforming_class
            for transforming_class in transforming_classes
        }

    def initialize_dataframe(self):
        """Inits the main dataframe with base features."""
        # TODO: check if the set of columns in dataframe after initialiation is exactly
        # the set of base features.
        raise NotImplementedError

    def load_raw_data(self):
        """Loads data from raw csv files and save them to self.raw_data."""
        raise NotImplementedError

    def load_dataframe(self):
        """Loads the dataframe from disk with appropriate types."""
        parse_dates = [
            feature
            for feature, metadata in self.feature_metadata.items()
            if metadata.dtype == feature_manager_pb2.DATETIME
        ]
        self.dataframe = pd.read_csv(
            self.dataset_path,
            dtype={
                feature: PANDAS_DTYPE_MAPPING[metadata.dtype]
                for feature, metadata in self.feature_metadata.items()
                if metadata.dtype != feature_manager_pb2.DATETIME
            },
            parse_dates=parse_dates,
        )

    def load_transformers(self):
        """Loads transformers from pickle."""
        with open(self.transformer_path, "rb") as pickle_file:
            self.transformer_dict = pickle.load(pickle_file)

    def save_dataframe(self):
        """Saves the dataframe to disk."""
        if not self.dataset_path.parent.exists():
            self.dataset_path.parent.mkdir(parents=True)
        logger.info(f"Saving dataframe to {self.dataset_path}")
        self.dataframe.to_csv(self.dataset_path, index=False)

    def save_transformers(self):
        """Saves the transformers to disk."""
        if not self.dataset_path.parent.exists():
            self.dataset_path.parent.mkdir(parents=True)
        logger.info(f"Saving transformers to {self.transformer_path}")
        with open(self.transformer_path, "wb") as pickle_file:
            pickle.dump(self.transformer_dict, pickle_file)

    def compute_feature(self, feature_name: str, is_training: bool = True) -> None:
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
        if not self.transforming_class_by_feature_name:
            self._get_transforming_class_by_name()
        self._compute_feature(feature_name, is_training=is_training)

    def _compute_feature(self, feature_name: str, is_training: bool = True) -> None:
        if self.dataframe is None:
            raise NotImplementedError(
                "self.dataframe must be initialized in self.initialize_dataframe()"
            )
        transforming_class = self.transforming_class_by_feature_name[feature_name]
        transformer = self.transformer_dict.get(feature_name)
        transformer_object = transforming_class(
            dependencies=self.config_helper.find_dependencies(feature_name),
            raw_data=self.raw_data,
        )
        series = transformer_object._transform(self.dataframe, transformer)
        if is_training:
            self.transformer_dict[feature_name] = transformer_object.transformer
        dtype = self.feature_metadata[feature_name].dtype
        if dtype == feature_manager_pb2.DATETIME:
            self.dataframe.loc[:, feature_name] = pd.to_datetime(series)
        else:
            self.dataframe.loc[:, feature_name] = pd.Series(
                series, dtype=PANDAS_DTYPE_MAPPING[dtype]
            )

    def compute_all_transforming_features(
        self, transforming_features: Union[List[str], None] = None, is_training=True
    ):
        """Computes all transforming feature.

        Should be done occasionally. After the first time this method is called, it's
        expected that features are updated one by one or in a set of few features.
        """
        if transforming_features is None:
            transforming_features = self.config_helper.transforming_features
        for feature_name in transforming_features:
            self.compute_feature(feature_name, is_training=is_training)

    def update_feature(self, feature_name: str):
        """Updates one feature in the dataframe.

        If this is an existing feature, all of its dependents should be computed.
        """
        features_to_compute = [feature_name]
        features_to_compute.extend(self.config_helper.find_dependents(feature_name))
        for feature_name in features_to_compute:
            self._compute_feature(feature_name)

    def run_all(self):
        self.load_raw_data()
        self.initialize_dataframe()
        self.compute_all_transforming_features()
        self.save_dataframe()
        self.save_transformers()

    def transform_new_samples(self, raw_data_samples, transforming_features):
        self.set_raw_data(raw_data_samples)
        self.initialize_dataframe()
        self.load_transformers()
        if transforming_features is None:
            # If transforming features are not specified, get all transforming features.
            transforming_features_and_dependencies = (
                self.config_helper.transforming_features
            )
        else:
            # Find all dependencies for the transforming features. The base features
            # should be excluded since they are pre-computed in
            # self.initialize_dataframe()
            transforming_features_and_dependencies = [
                feature
                for feature in self.config_helper.get_dependencies_recursively(
                    list(transforming_features)
                )
                if feature not in self.config_helper.base_features
            ]
        self.compute_all_transforming_features(
            transforming_features_and_dependencies, is_training=False
        )
        return self.dataframe

    def set_raw_data(self, raw_data_samples: Any):
        NotImplementedError

    def extract_dataframe(
        self, features_to_select: List[str], filters: Optional[List[str]] = None
    ):
        """Extracts a subset of columns and a subset of rows in the dataframe.

        The whole idea of Feature Manager is to create a big dataframe with as many
        columns and rows as we want, which helps store the whole information in one
        place. All training/validation/test datasets are stored in this big table and
        are then extracted later per use case.

        To extract columns, we can simply use a list of column names. To extract rows,
        we can use a set of boolen filters. Training, validation and test data should
        be extracted using these filters.

        Args:
            features_to_select: A list of column names to be selected.
            filters: A list of boolen filters used to extract rows.
        """
        assert (
            self.dataframe is not None
        ), "You may forget to load the dataframe, use method load_dataframe first."

        if filters is None:
            return self.dataframe[features_to_select]
        query_to_filter = " and ".join(filters)
        return self.dataframe.query(query_to_filter)[features_to_select]


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
        self.transformer = None

    def _transform(self, dataframe, transformer=None):
        logger.info(f"Computing feature {self.name} in pandas ...")
        # This is to make sure that _transform_ methods do not use any columns other
        # than dependencies.
        df = dataframe[self.dependencies]
        self.transformer = transformer
        if transformer is None:
            self.fit(dataframe)
        return self.transform(df)

    def fit(self, df):
        # The subclasses need to override this method if there is a need to fit the
        # transformer to data.
        self.transformer = None

    @abstractmethod
    def transform(self, df):
        raise NotImplementedError
