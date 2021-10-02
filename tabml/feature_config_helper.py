import copy
from typing import Dict, List, Union

from tabml.config_helpers import parse_feature_config
from tabml.schemas.feature_config import BaseFeature, DType, TransformingFeature
from tabml.utils.utils import check_uniqueness


class FeatureMetadata:
    def __init__(
        self, index: int, dtype: DType, dependencies: Union[List, None] = None
    ):
        self.index = index
        self.dependents: List[str] = []
        self.dtype = dtype
        self.dependencies = dependencies or []

    @classmethod
    def from_base_feature(cls, feature: BaseFeature):
        # All base features are considered to have index 0.
        return cls(index=0, dtype=feature.dtype)

    @classmethod
    def from_transforming_feature(cls, feature: TransformingFeature):
        return cls(
            index=feature.index, dtype=feature.dtype, dependencies=feature.dependencies
        )


class FeatureConfigHelper:
    """A config helper class for feature manager.

    Attributes:
        config:
            A feature_config object parsed from a yaml file.
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

    def __init__(self, config_path: str):
        self.config = parse_feature_config(config_path)
        self.raw_data_dir = self.config.raw_data_dir
        self.dataset_name = self.config.dataset_name
        self.base_features = [feature.name for feature in self.config.base_features]
        self.transforming_features = [
            transforming_feature.name
            for transforming_feature in self.config.transforming_features
        ]
        self.all_features = self.base_features + self.transforming_features
        self._validate()
        self.feature_metadata: Dict[str, FeatureMetadata] = {}
        self._build_feature_metadata()

    def _validate(self):
        self._validate_indexes()
        self._validate_dependency()
        check_uniqueness(self.all_features)

    def _validate_indexes(self):
        """Checks if indexes are positive and monotonically increasing."""
        indexes = [
            transforming_feature.index
            for transforming_feature in self.config.transforming_features
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

    def _validate_dependency(self):
        """Checks if all dependencies of a transforming feature exist."""
        # Start from base_features then gradually add transforming_features.
        features_so_far = self.base_features.copy()
        for feature in self.config.transforming_features:
            for dependency in feature.dependencies:
                assert (
                    dependency in features_so_far
                ), "Feature {} depends on feature {} that is undefined.".format(
                    feature.name, dependency
                )
            features_so_far.append(feature.name)

    def _build_feature_metadata(self):
        for feature in self.config.base_features:
            self.feature_metadata[feature.name] = FeatureMetadata.from_base_feature(
                feature
            )

        for feature in self.config.transforming_features:
            self.feature_metadata[
                feature.name
            ] = FeatureMetadata.from_transforming_feature(feature)
            for dependency in feature.dependencies:
                self.feature_metadata[dependency].dependents.append(feature.name)

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
            queue.extend(self.get_direct_dependencies(feature))
        return self.sort_features(seen)

    def get_direct_dependencies(self, feature_name: str) -> List[str]:
        return self.feature_metadata[feature_name].dependencies

    def sort_features(self, feature_names: List[str]) -> List[str]:
        return sorted(feature_names, key=lambda x: self.feature_metadata[x].index)

    def get_dependents_recursively(self, feature_name: str) -> List[str]:
        """Finds all features that are directly/indirectly dependents of a feature.

        This is necessary when we want to update one feature. All of its dependents also
        need to be updated. The list of returning features is required to be in the
        order determined by their indexes in the config.

        Notes:
            * If "b" is a dependency of "a" then "a" is a dependent of "b".
            * If "a" is a dependent of "b" and "b" is a dependent of "c" then "a" is a
              dependent of "c".
        """
        # BFS to find all dependents.
        dependents: List[str] = []
        queue = self.feature_metadata[feature_name].dependents
        while queue:
            feature = queue.pop(0)
            if feature in dependents:
                continue
            dependents.append(feature)
            queue.extend(self.feature_metadata[feature].dependents)
        return self.sort_features(dependents)

    def extract_config(self, selected_features: List[str]):
        """Creates a minimum valid config that contains all selected_features.

        Returns a feature_config object with only a subset of transforming features and
        all of their dependencies.

        NOTE: all base features will be in the extracted config; they were created in
        the raw data loading step.

        Args:
            selected_features: a list of selected features.

        """
        self._validate_features(selected_features)

        new_config = copy.deepcopy(self.config)
        required_features = self.get_dependencies_recursively(
            features=selected_features
        )
        minimum_transforming_features = [
            transforming_feature
            for transforming_feature in self.config.transforming_features
            if transforming_feature.name in required_features
        ]
        new_config.transforming_features = minimum_transforming_features

        return new_config

    def _validate_features(self, features: List[str]):
        """Checks if all features are in the config.

        Raises:
            ValueError if selected_features contains an unknown features (not in the
            original config).
        """
        invalid_features = [
            feature for feature in features if feature not in self.all_features
        ]
        if invalid_features:
            raise ValueError(
                f"Features {invalid_features} are not in the original config."
            )
