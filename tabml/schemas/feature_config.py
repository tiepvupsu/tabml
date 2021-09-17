from enum import Enum
from typing import List

import pydantic


class DType(Enum):
    INT32 = "INT32"
    BOOL = "BOOL"
    FLOAT = "FLOAT"
    INT64 = "INT64"
    STRING = "STRING"
    DOUBLE = "DOUBLE"
    # data and time types https://docs.python.org/3/library/datetime.html
    DATE = "DATE"
    TIME = "TIME"
    DATETIME = "DATETIME"


class BaseFeature(pydantic.BaseModel):
    name: str
    dtype: DType


class TransformingFeature(pydantic.BaseModel):
    # feature name, need to be unique in the FeatureConfig.
    name: str

    # index of the feature in the dataset. Indexes should be unique, positive and
    # monotonically increasing. Indexes are used to determine the feature order in the
    # whole dataset. Base features are added to the dataset first, then
    # transforming_features in the ascending order of indexes.
    index: int

    # dependencies is a list of features that are required to compute this feature.
    # This field will be used when users want to re-compute one feature. All features
    # depending on this re-computed feature are also required to be updated.
    # NOTE on terminology: If feature "a" depends on feature "b" then we call "b" is
    # one of dependencies of "a", and "a" is a dependent of "b".
    dtype: DType

    dependencies: List[str] = []


class FeatureConfig(pydantic.BaseModel):
    # directory of raw data files
    raw_data_dir: str

    # name of the dataframe to store features, also the name of the csv dataframe file.
    # The csv file path will be raw_data_dir / features / dataset_name + ".csv".
    dataset_name: str

    # base_features are features that are not dependent on any features.
    # These features are usually created right after the data cleaning step.
    base_features: List[BaseFeature]

    # transforming_features are those dependent on base features and/or other
    # transforming features. Note that the term "feature" here only apply to columns in
    # the final dataframe saved in dataset_name. Data taken from additional dataframes
    # is not considered as features.
    transforming_features: List[TransformingFeature]
