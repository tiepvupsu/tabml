import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from tabml.config_helpers import parse_feature_config, parse_pipeline_config
from tabml.experiment_manager import ExperimentManager
from tabml.feature_config_helper import FeatureConfigHelper
from tabml.schemas.feature_config import DType, FeatureConfig

from tabml.schemas.bundles import FeatureBundle
from tabml.schemas.pipeline_config import ModelBundle
from tabml.utils.logger import logger
from tabml.utils.utils import (
    check_uniqueness,
    load_pickle,
    mkdir_if_needed,
    return_or_load,
)

PANDAS_DTYPE_MAPPING = {
    DType.BOOL: "bool",
    DType.INT32: "int32",
    DType.INT64: "int64",
    DType.STRING: "str",
    DType.FLOAT: "float32",
    DType.DOUBLE: "float64",
    # DATETIME will be converted to datetime parse_date https://tinyurl.com/y4waw6np
    DType.DATETIME: "datetime64[ns]",
}

FEATURE_BUNDLE_FILENAME = "feature_bundle.pickle"


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
        custom_feature_bundle_path:
            pickle path to save both feature config and transformer
    """

    def __init__(
        self,
        config: Union[str, Path, FeatureConfig],
        custom_feature_bundle_path: Union[str, None] = None,
    ):
        _config = return_or_load(config, FeatureConfig, parse_feature_config)

        self.config_helper = FeatureConfigHelper(_config)
        self.raw_data_dir = self.config_helper.raw_data_dir
        self.dataset_name = self.config_helper.dataset_name
        self.feature_metadata = self.config_helper.feature_metadata
        self.dataset_path = self.get_dataset_path()
        self.features_in_config: List[str] = self.config_helper.all_feature_names
        self.raw_data: Dict[str, Any] = {}
        self.dataframe: pd.DataFrame = pd.DataFrame()
        self.transforming_class_by_feature_name: Dict[str, Any] = {}
        self.transformer_dict: Dict[str, Any] = {}
        self.feature_bundle_path = (
            custom_feature_bundle_path or self.get_feature_bundle_path()
        )

    @classmethod
    def from_feature_bundle(cls, feature_bundle: Union[str, FeatureBundle]):
        _feature_bundle = return_or_load(feature_bundle, FeatureBundle, load_pickle)
        feature_config = _feature_bundle.feature_config
        fm = cls(feature_config)
        fm.transformer_dict = _feature_bundle.transformers
        return fm

    def save_feature_bundle(self):
        data = FeatureBundle(
            feature_config=self.config_helper.config, transformers=self.transformer_dict
        )
        mkdir_if_needed(self.dataset_path.parent)
        save_path = self.feature_bundle_path
        logger.info(f"Saving feature bundle to {save_path}")
        with open(save_path, "wb") as pickle_file:
            pickle.dump(data, pickle_file)

    def get_dataset_path(self):
        return Path(self.raw_data_dir) / "features" / f"{self.dataset_name}.csv"

    def get_feature_bundle_path(self):
        return Path(self.raw_data_dir) / "features" / FEATURE_BUNDLE_FILENAME

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
            if metadata.dtype == DType.DATETIME
        ]
        self.dataframe = pd.read_csv(
            self.dataset_path,
            dtype={
                feature: PANDAS_DTYPE_MAPPING[metadata.dtype]
                for feature, metadata in self.feature_metadata.items()
                if metadata.dtype != DType.DATETIME
            },
            parse_dates=parse_dates,
        )

    def save_dataframe(self):
        """Saves the dataframe to disk."""
        if not self.dataset_path.parent.exists():
            self.dataset_path.parent.mkdir(parents=True)
        logger.info(f"Saving dataframe to {self.dataset_path}")
        self.dataframe.to_csv(self.dataset_path, index=False)

    def compute_transforming_feature(
        self, feature_name: str, is_training: bool = True
    ) -> None:
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
        self._compute_transforming_feature(feature_name, is_training=is_training)

    def _compute_transforming_feature(
        self, feature_name: str, is_training: bool = True
    ) -> None:
        if self.dataframe is None:
            raise NotImplementedError(
                "self.dataframe must be initialized in self.initialize_dataframe()"
            )
        transforming_class = self.transforming_class_by_feature_name[feature_name]
        transformer = self.transformer_dict.get(feature_name)
        transformer_object = transforming_class(
            dependencies=self.config_helper.get_direct_dependencies(feature_name),
            raw_data=self.raw_data,
        )
        series = transformer_object._transform(self.dataframe, transformer)
        if is_training:
            self.transformer_dict[feature_name] = transformer_object.transformer

        self._update_dataframe(feature_name, series)

    def _update_dataframe(self, feature_name, series):
        dtype = self.feature_metadata[feature_name].dtype
        if dtype == DType.DATETIME:
            self.dataframe.loc[:, feature_name] = pd.to_datetime(series)
        else:
            self.dataframe.loc[:, feature_name] = pd.Series(
                series, dtype=PANDAS_DTYPE_MAPPING[dtype]
            )

    def compute_transforming_features(
        self, transforming_features: Union[List[str], None] = None, is_training=True
    ):
        """Computes transforming features.

        Should be done occasionally. After the first time this method is called, it's
        expected that features are updated one by one or in a set of few features.
        """
        if transforming_features is None:
            transforming_features = self.config_helper.transforming_feature_names
        for feature_name in transforming_features:
            self.compute_transforming_feature(feature_name, is_training=is_training)

    def update_transforming_feature(self, feature_name: str):
        """Updates one transforming feature in the dataframe.

        If this is an existing feature, all of its dependents should be computed.
        """
        features_to_compute = [feature_name]
        features_to_compute.extend(
            self.config_helper.get_dependents_recursively(feature_name)
        )
        for _feature_name in features_to_compute:
            self._compute_transforming_feature(_feature_name)

    def run_all(self):
        self.load_raw_data()
        self.initialize_dataframe()
        self.compute_transforming_features()
        self.save_dataframe()
        self.save_feature_bundle()

    def compute_prediction_features(
        self, prediction_feature_names: Union[List[str], None] = None
    ):
        self.load_dataframe()
        if prediction_feature_names is None:
            prediction_feature_names = self.config_helper.prediction_feature_names
        self._validate_prediction_feature_names(prediction_feature_names)
        for prediction_feature_name in prediction_feature_names:
            self._compute_prediction_feature(prediction_feature_name)
        self.save_dataframe()

    def _compute_prediction_feature(self, prediction_feature_name: str):
        # import here to avoid circular imports
        from tabml.model_wrappers import initialize_model_wrapper

        logger.info(f"Computing prediction feature {prediction_feature_name} ...")
        metadata = self.feature_metadata[prediction_feature_name]
        model_bundle = metadata.model_bundle
        _model_bundle = return_or_load(model_bundle, ModelBundle, load_pickle)
        _model_wrapper = initialize_model_wrapper(model_bundle)
        features_to_pred_model = (
            _model_bundle.pipeline_config.data_loader.features_to_model
        )
        preds = _model_wrapper.predict(self.dataframe[features_to_pred_model])
        self._update_dataframe(prediction_feature_name, preds)

    def _validate_prediction_feature_names(self, prediction_feature_names: List[str]):
        undefined_features = [
            feature
            for feature in prediction_feature_names
            if feature not in self.config_helper.prediction_feature_names
        ]
        if undefined_features:
            raise ValueError(
                f"Features {undefined_features} are not defined in feature config."
            )

    def transform_new_samples(self, raw_data_samples, transforming_features):
        self.set_raw_data(raw_data_samples)
        self.initialize_dataframe()
        if transforming_features is None:
            # If transforming features are not specified, get all transforming features.
            transforming_features_and_dependencies = (
                self.config_helper.transforming_feature_names
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
                if feature not in self.config_helper.base_feature_names
            ]
        self.compute_transforming_features(
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
        pass


@dataclass
class ModelInferenceWithPreprocessedData:
    model_path: str = ""
    pipeline_config_path: Union[str, None] = None

    def __post_init__(self):
        from tabml.model_wrappers import load_or_train_model  # isort:skip

        config = _get_config(self.pipeline_config_path, self.model_path)
        self.model_wrapper = load_or_train_model(
            self.model_path, self.pipeline_config_path
        )
        self.features_to_model = list(config.data_loader.features_to_model)

    def predict(self, data):
        return self.model_wrapper.predict(data[self.features_to_model])


# TODO: move this and a similar function in inference.py to a common place
def _get_config(pipeline_config_path, model_path):
    pipeline_config_path = (
        pipeline_config_path
        or ExperimentManager.get_config_path_from_model_path(model_path)
    )
    return parse_pipeline_config(pipeline_config_path)
