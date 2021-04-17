from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tabml import data_processing
from tabml.feature_manager import BaseFeatureManager, BaseTransformingFeature
from tabml.utils.git import get_git_repo_dir


class FeatureManager(BaseFeatureManager):
    def __init__(self, pb_config_path):
        super().__init__(pb_config_path)

    def _get_base_transforming_class(self):
        return BaseHousingTransformingFeature

    def load_raw_data(self):
        raw_data_dir = Path(self.raw_data_dir)
        full_df = pd.read_csv(raw_data_dir / "housing.csv")
        self.raw_data["full"] = full_df

    def initialize_dataframe(self):
        self.dataframe = self.raw_data["full"][
            [
                "median_house_value",
                "housing_median_age",
                "total_rooms",
                "population",
                "total_bedrooms",
                "households",
                "median_income",
                "ocean_proximity",
                "latitude",
                "longitude",
            ]
        ]


class BaseHousingTransformingFeature(BaseTransformingFeature):
    pass


class FeatureAge(BaseHousingTransformingFeature):
    name = "is_train"

    def transform(self, df):
        np.random.seed(42)
        random_array = np.random.rand(len(df))
        validation_size = 0.1
        return random_array > validation_size


class FeatureHousingMedianAge(BaseHousingTransformingFeature):
    name = "scaled_housing_median_age"

    def transform(self, df):
        return data_processing.fit_train_transform_all(
            whole_df=df,
            input_columns=["housing_median_age"],
            training_filters=["is_train"],
            transformer=StandardScaler(),
        ).reshape(-1)


def _scale_clean_transform(df, input_columns):
    return data_processing.fit_train_transform_all(
        whole_df=df,
        input_columns=input_columns,
        training_filters=["is_train"],
        transformer=Pipeline(
            [
                ("clip_outlier", data_processing.BoxplotOutlierClipper()),
                ("std_scaler", StandardScaler()),
            ]
        ),
    ).reshape(-1)


class FeatureScaledCleanTotalRooms(BaseHousingTransformingFeature):
    """Feature created by applying box plot to fix outliers then StandardScaler."""

    name = "scaled_clean_total_rooms"

    def transform(self, df):
        return _scale_clean_transform(df, ["total_rooms"])


class FeatureScaledCleanTotalBedrooms(BaseHousingTransformingFeature):
    name = "scaled_clean_total_bedrooms"

    def transform(self, df):
        return _scale_clean_transform(df, ["total_bedrooms"])


class FeatureScaledCleanPopulation(BaseHousingTransformingFeature):
    name = "scaled_clean_population"

    def transform(self, df):
        return _scale_clean_transform(df, ["population"])


class FeatureScaledCleanHouseholds(BaseHousingTransformingFeature):
    name = "scaled_clean_households"

    def transform(self, df):
        return _scale_clean_transform(df, ["households"])


class FeatureScaledCleanMedianIncome(BaseHousingTransformingFeature):
    name = "scaled_clean_median_income"

    def transform(self, df):
        return _scale_clean_transform(df, ["median_income"])


class FeatureLog10MedianHouseValue(BaseHousingTransformingFeature):
    name = "log10_median_house_value"

    def transform(self, df):
        return np.log10(df["median_house_value"])


class FeatureBucketizedLatitude(BaseHousingTransformingFeature):
    name = "bucketized_latitude"

    def transform(self, df):
        return pd.qcut(df["latitude"], q=10, labels=False)


class FeatureBucketizedLongitude(BaseHousingTransformingFeature):
    name = "bucketized_longitude"

    def transform(self, df):
        return pd.qcut(df["longitude"], q=10, labels=False)


class FeatureBucketizedLatitudeXBucketizedLongitude(BaseHousingTransformingFeature):
    name = "bucketized_latitude_X_bucketized_longitude"

    def transform(self, df):
        # TODO make feature cross more generic
        return data_processing.cross_columns(
            df, cols=["bucketized_latitude", "bucketized_longitude"]
        )


class FeatureHasdedBucketizedLatitudeXBucketizedLongitude(
    BaseHousingTransformingFeature
):
    name = "hashed_bucketized_latitude_X_bucketized_longitude"

    def transform(self, df):
        hash_bucket_size = 256
        return df["bucketized_latitude_X_bucketized_longitude"].apply(
            lambda x: data_processing.hash_modulo(x, hash_bucket_size)
        )


class FeatureEncodedOceanProximity(BaseHousingTransformingFeature):
    name = "encoded_ocean_proximity"

    def transform(self, df):
        return LabelEncoder().fit_transform(df["ocean_proximity"])


def run():
    pb_config_path = (
        Path(get_git_repo_dir()) / "tabml/housing/configs/feature_config.pb"
    )
    fm = FeatureManager(pb_config_path)
    fm.run_all()


if __name__ == "__main__":
    run()
