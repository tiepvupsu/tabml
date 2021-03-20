import copy
from pathlib import Path
from typing import Any, Collection, Dict, List, Set, Union

import pandas as pd

from tabml.protos import feature_manager_pb2
from tabml.utils.logger import logger
from tabml.utils.pb_helpers import parse_feature_config_pb

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


class BaseFeatureManager:
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
        dataframe: The main pandas dataframe.
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
        self.features_in_config = self.config_helper.all_features
        self.raw_data: Dict[str, Any] = {}
        self.dataframe = pd.DataFrame()
        self.base_transforming_feature_class = self._get_base_transforming_class()
        self.transforming_class_by_feature_name = self._get_transforming_class_by_name()

    def _get_base_transforming_class(self):
        # should be implemented in subclasses.
        raise NotImplementedError

    def load_raw_data(self):
        """Loads all data required to build the dataset."""
        # should be implemented in subclasses.
        raise NotImplementedError

    def load_dataframe(self):
        """Loads the dataframe from disk."""
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

    def save_dataframe(self):
        """Saves the dataframe to disk."""
        if not self.dataset_path.parent.exists():
            self.dataset_path.parent.mkdir(parents=True)
        logger.info(f"Saving dataframe to {self.dataset_path}")
        self.dataframe.to_csv(self.dataset_path, index=False)

    def initialize_dataframe(self):
        """Initializes the dataframe with base features."""
        # should be implemented in subclasses.
        raise NotImplementedError

    def load_or_initialize_dataframe(self):
        """Loads dataframe if it exists, initializes otherwise."""
        if self.dataset_path.exists():
            self.load_dataframe()
        else:
            self.initialize_dataframe()

    def _get_transforming_class_by_name(self) -> Dict[str, Any]:
        if self.base_transforming_feature_class is None:
            raise NotImplementedError
        transforming_classes = self.base_transforming_feature_class.__subclasses__()
        _check_uniqueness(
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
        )._transform(self.dataframe)
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

    def update_features(self, feature_names: List[str]):
        """Updates multiple features at once."""
        features_to_compute = self.config_helper.append_dependents(feature_names)
        for feature_name in features_to_compute:
            self._compute_feature(feature_name)

    def update_and_save_features(self, feature_names: List[str]):
        """Updates multiple features and saves it."""
        self.load_raw_data()
        self.load_dataframe()
        self.update_features(feature_names)
        self.save_dataframe()

    def compute_all_transforming_features(self):
        """Computes all transforming feature.

        Should be done occasionally. After the first time this method is called, it's
        expected that features are updated one by one or in a set of few features.
        """
        for idx, feature_name in enumerate(self.config_helper.transforming_features):
            self.compute_feature(feature_name)
            if idx % 5 == 0:
                self.dataframe = self.dataframe.checkpoint()

        self.dataframe = self.dataframe.select(*self.feature_metadata.keys())

    def run_all(self, should_save_dataframe: bool = True):
        """Runs end-to-end data pipeline.

        Args:
            should_save_dataframe: save dataframe or not (default True).
        """
        self.load_raw_data()
        self.initialize_dataframe()
        self.compute_all_transforming_features()
        if should_save_dataframe:
            self.save_dataframe()

    def _get_features_in_dataframe(self) -> List[str]:
        return self.dataframe.columns.tolist()

    def compute_new_features(self):
        """Makes the update after adding more features to the config.

        This method looks for features that are in config but not in stored dataframe
        and computes these new features.
        """
        self.load_raw_data()
        self.load_or_initialize_dataframe()

        features_to_compute = [
            feature_name
            for feature_name in self.features_in_config
            if feature_name not in self._get_features_in_dataframe()
        ]
        if features_to_compute:
            for feature in features_to_compute:
                self.compute_feature(feature)
            self.save_dataframe()
        else:
            logger.info(
                "There is no new feature, if you want to force updating features, "
                "use update_features() method instead."
            )

    def cleanup(self):
        """Cleans up old features in the dataframe.

        When one or more features need to be cleaned, developers simply remove them from
        the feature manager config and run this method.
        """
        self.load_dataframe()
        features_to_clean = [
            feature_name
            for feature_name in self._get_features_in_dataframe()
            if feature_name not in self.features_in_config
        ]

        # Logically, when deleting one feature, we need to make sure there will be no
        # orphan dependents. Fortunately, there is no need to check depdencency here
        # since it should be already validated by
        # FeatureConfigHelper._validate_dependency()

        # While `inplace=True` can be set here, it's discouraged by the pandas team.
        # See more at https://github.com/pandas-dev/pandas/issues/16529.
        self.dataframe = self.dataframe.drop(columns=features_to_clean)

    def extract_dataframe(
        self, features_to_select: List[str], filters: Union[List[str], None]
    ):
        """Extracts a subset of columns and a subset of rows in the dataframe.

        The whole idea of Feature Manager is to create a big dataframe with as many
        columns and rows as we want, which helps store the whole information. All
        training/validation/test datasets are stored in this big table and are then
        extracted later per use case.

        To extract columns, we can simply use a list of column names. To extract rows,
        we can use a set of boolen filters to get columns we want. Training, validation
        and test data should be extracted using these filters.

        Args:
            features_to_select: A list of column names to be selected.
            filters: A list of names of boolen filter used to extract rows.
        """
        assert (
            self.dataframe is not None
        ), "You may forget to load the dataframe, use method load_dataframe first."

        if filters is None:
            return self.dataframe[features_to_select]
        query_to_filter = " and ".join(filters)
        return self.dataframe.query(query_to_filter)[features_to_select]


class BaseTransformingFeature:
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

    def transform(self, df):
        raise NotImplementedError("Must be implemented in subclasses.")


class _Feature:
    def __init__(self, index: int, dtype: str, dependencies: Union[List, None] = None):
        self.index = index
        self.dependents: List[str] = []
        self.dtype = dtype
        if dependencies is None:
            self.dependencies = []
        else:
            self.dependencies = dependencies


class FeatureConfigHelper:
    """A config helper class for feature manager.

    Attributes:
        _config:
            A protobuf object parsed from a pb text file.
        raw_data_dir:
            A string of directory to raw data.
        dataset_name:
            A string of the dataset name
        base_features:
            A list of base feature names in the config.
        transforming_features:
            A list of transforming feature names in the config.
        all_features:
            A list of all features in the config.
        feature_metadata:
            A dict of {feature_name: (index, dtype, list of its dependents)}.
            This is useful when finding all dependents of one feature.
    """

    def __init__(self, pb_config_path: str):
        self._config = parse_feature_config_pb(pb_config_path)
        self.raw_data_dir = self._config.raw_data_dir
        self.dataset_name = self._config.dataset_name
        self.base_features = [feature.name for feature in self._config.base_features]
        self.transforming_features = [
            transforming_feature.name
            for transforming_feature in self._config.transforming_features
        ]
        self.all_features = self.base_features + self.transforming_features
        self._validate()
        self.feature_metadata: Dict[str, _Feature] = {}
        self._build_feature_metadata()

    def _validate(self):
        self._validate_indexes()
        self._validate_uniqueness()
        self._validate_dependency()

    def _validate_indexes(self):
        """Checks if indexes are positive and monotonically increasing."""
        indexes = [
            transforming_feature.index
            for transforming_feature in self._config.transforming_features
        ]
        if not (
            indexes[0] > 0
            and all(
                [
                    index_i < index_ip1
                    for (index_i, index_ip1) in zip(indexes[:-1], indexes[1:])
                ]
            )
        ):
            raise ValueError(
                "Feature indexes must be a list of increasing positive integers. "
                f"Got indexes = {indexes}"
            )

    def _validate_uniqueness(self):
        _check_uniqueness(self.all_features)

    def _validate_dependency(self):
        """Checks if all dependencies of a transforming feature exists."""
        # initialize from base_features then gradually adding transforming_feature
        features_so_far = self.base_features.copy()
        for feature in self._config.transforming_features:
            for dependency in feature.dependencies:
                assert (
                    dependency in features_so_far
                ), "Feature {} depends on feature {} that is undefined.".format(
                    feature.name, dependency
                )
            features_so_far.append(feature.name)

    def _build_feature_metadata(self):
        for feature in self._config.base_features:
            # all base features are considered to have index 0
            self.feature_metadata[feature.name] = _Feature(index=0, dtype=feature.dtype)

        for feature in self._config.transforming_features:
            self.feature_metadata[feature.name] = _Feature(
                index=feature.index,
                dtype=feature.dtype,
                dependencies=feature.dependencies,
            )
            for dependency in feature.dependencies:
                self.feature_metadata[dependency].dependents.append(feature.name)

    def sort_features(self, feature_names: List[str]) -> List[str]:
        return sorted(feature_names, key=lambda x: self.feature_metadata[x].index)

    def find_dependencies(self, feature_name: str) -> List[str]:
        return self.feature_metadata[feature_name].dependencies

    def get_dependencies_recursively(self, features: List[str]) -> List[str]:
        """Gets all dependencies of a list of features recursively.

        The input list should also be in the result.
        """
        queue = copy.copy(features)
        seen: List[str] = []
        while queue:
            feature = queue.pop(0)
            if feature in seen:
                continue
            seen.append(feature)
            queue.extend(self.find_dependencies(feature))
        return self.sort_features(seen)

    def find_dependents(self, feature_name: str) -> List[str]:
        """Finds all features that are directly/indirectly dependent on a feature.

        This is necessary when we want to update one feature. All of its dependents also
        need to be updated. The list of returning features is required to be in the
        order determined by their indexes in the proto config.

        Notes:
            * If "b" is a dependency of "a" then "a" is a dependent of "b".
            * If "a" is a dependent of "b" and "b" is a dependent of "c" then "a" is a
              dependent of "c".
        """
        # BFS to find all dependents
        dependents: List[str] = []
        queue = self.feature_metadata[feature_name].dependents
        while queue:
            feature = queue.pop(0)
            if feature in dependents:
                continue
            dependents.append(feature)
            queue.extend(self.feature_metadata[feature].dependents)
        return self.sort_features(dependents)

    def append_dependents(self, feature_names: List[str]) -> List[str]:
        """Finds all dependents of a list of features then appends them to the list.

        This will be used when multiple features need to be updated or remove. For the
        first case, all dependents also need to be updated. For the second case, it's
        required that there is no dependent in the remaining features.

        NOTE: the results should also contain the input feature_names.
        """
        all_features = feature_names
        for feature_name in feature_names:
            all_features.extend(self.find_dependents(feature_name))

        return self.sort_features(list(set(all_features)))

    def extract_config(self, selected_features: List[str]):
        """Creates a minimum valid config that contains all selected_features.

        Returns a protobuf with only a subset of transforming features and all of their
        dependencies.

        NOTE: all base features will be in the extracted config; they were created in
        the raw data loading step.

        Args:
            selected_features: a list of selected features.

        Raises:
            ValueError if selected_features contains an unknown features (not in the
            original config).
        """
        invalid_features = [
            feature for feature in selected_features if feature not in self.all_features
        ]
        if invalid_features:
            raise ValueError(
                f"Features {invalid_features} are not in the original config."
            )
        all_relevant_features = self.get_dependencies_recursively(
            features=selected_features
        )
        new_pb = copy.deepcopy(self._config)
        # we can't deriectly assign a list to a protobuf repeated field
        # https://tinyurl.com/y4m86cc4
        del new_pb.transforming_features[:]
        new_pb.transforming_features.extend(
            [
                transforming_feature
                for transforming_feature in self._config.transforming_features
                if transforming_feature.name in all_relevant_features
            ]
        )

        return new_pb


def _check_uniqueness(items: Collection) -> None:
    """Checks if an array containing unique elements.

    Args:
        items: A list of objects.

    Returns:
        Does not return anything. If this function passes, it means that all objects
        are unique.

    Raises:
        Assertion error with list of duplicate objects.
    """
    seen_items: Set[Any] = set()
    duplicates = set()
    for item in items:
        if item in seen_items:
            duplicates.add(item)
        seen_items.add(item)
    assert not duplicates, f"There are duplicate objects in the list: {duplicates}."
