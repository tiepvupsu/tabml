from enum import Enum
from pathlib import Path
from typing import List, Union

import pydantic

from tabml.schemas.pipeline_config import ModelBundle


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


class Feature(pydantic.BaseModel):
    name: str
    dtype: DType


class TransformingFeature(pydantic.BaseModel):
    name: str

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
    dtype: DType
    model_bundle: Union[str, Path, ModelBundle] = ""


class LegacyFeatureConfig(pydantic.BaseModel):
    # Directory of raw data files.
    raw_data_dir: str

    # Name of the dataframe to store features, also the name of the csv dataframe file.
    # The csv file path will be raw_data_dir / features / dataset_name + ".csv".
    dataset_name: str

    # Base features are features that are not dependent on any features.
    # These features are usually created right after the data cleaning step.
    base_features: List[Feature]

    # Transforming features are those dependent on base features and/or other
    # transforming features. Note that the term "feature" here only apply to columns in
    # the final dataframe saved in dataset_name. Data taken from additional dataframes
    # is not considered as features.
    transforming_features: List[TransformingFeature] = []

    # Prediction features are those computed as predictions of another models. These
    # are useful for stacking models.
    # In the first version, transforming features those are dependent on prediction
    # features are not supported.
    # TODO: support transforming features those are dependent on prediction features.
    prediction_features: List[PredictionFeature] = []
