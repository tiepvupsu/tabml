from enum import Enum
from typing import List, Union
from pathlib import Path
from tabml.schemas.pipeline_config import ModelBundle

import pydantic


class DType(Enum):
    INT32 = "INT32"
    BOOL = "BOOL"
    FLOAT = "FLOAT"
    INT64 = "INT64"
    STRING = "STRING"
    DOUBLE = "DOUBLE"
    # Date and time types https://docs.python.org/3/library/datetime.html
    DATE = "DATE"
    TIME = "TIME"
    DATETIME = "DATETIME"


class BaseFeature(pydantic.BaseModel):
    name: str
    dtype: DType
    index: int = 0


class TransformingFeature(pydantic.BaseModel):
    name: str

    # The index of the feature in the dataset. Indexes should be unique, positive and
    # monotonically increasing. Indexes are used to determine the feature order in the
    # whole dataset. Base features are added to the dataset first, then
    # transforming_features and prediction_features in the ascending order of indexes.
    index: int

    dtype: DType

    # Dependencies is a list of features that are required to compute this feature.
    # This field will be used when users want to re-compute one feature. All features
    # depending on this re-computed feature are also required to be updated.
    # NOTE on terminology: If feature "a" depends on feature "b" then we call "b" is
    # one of dependencies of "a", and "a" is a dependent of "b".
    dependencies: List[str] = []


class PredictionFeature(pydantic.BaseModel):
    # Features that are predicted by a tabml model. These features could be used in
    # stacking models.
    name: str
    index: int
    dtype: DType
    model_bundle: Union[str, Path, ModelBundle] = ""


class FeatureConfig(pydantic.BaseModel):
    # Directory of raw data files.
    raw_data_dir: str

    # Name of the dataframe to store features, also the name of the csv dataframe file.
    # The csv file path will be raw_data_dir / features / dataset_name + ".csv".
    dataset_name: str

    # Base features are features that are not dependent on any features.
    # These features are usually created right after the data cleaning step.
    base_features: List[BaseFeature]

    # Transforming features are those dependent on base features and/or other
    # transforming features. Note that the term "feature" here only apply to columns in
    # the final dataframe saved in dataset_name. Data taken from additional dataframes
    # is not considered as features.
    transforming_features: List[TransformingFeature] = []

    # Prediction features are those computed as predictions of another models. These
    # are useful for stacking models.
    # In the first version, transforming features those are dependent on prediction
    # features are not supported.
    # TODO: support transforming features those are dependent on prediciton features.
    prediction_features: List[PredictionFeature] = []
