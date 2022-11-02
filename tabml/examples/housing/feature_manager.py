from typing import Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tabml import data_processing, datasets
from tabml.feature_manager import BaseFeatureManager, BaseTransformingFeature

DATA_URL = (
    "https://media.githubusercontent.com/media/tiepvupsu/tabml_data/master/"
    "california_housing/housing.csv"
)


class FeatureManager(BaseFeatureManager):
    def __init__(self, config, feature_bundle_path: Union[str, None] = None):
        super().__init__(config, feature_bundle_path)

    def _get_base_transforming_class(self):
        return BaseHousingTransformingFeature

    def load_raw_data(self):
        full_df = datasets.load_california_housing(self.raw_data_dir)["housing"]
        full_df["house_id"] = full_df.index
        self.raw_data["full"] = full_df

    def initialize_dataframe(self):
        self.dataframe = self.raw_data["full"][
            [
                "house_id",
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


class FeatureIsTrain(BaseHousingTransformingFeature):
    name = "is_train"

    def transform(self, df):
        # use 90% data for training
        return df["house_id"].apply(lambda x: data_processing.hash_modulo(x, 100) < 90)


class FeatureHousingMedianAge(BaseHousingTransformingFeature):
    name = "scaled_housing_median_age"

    def fit(self, df):
        self.transformer = StandardScaler()
        train_data = df.query("is_train")[["housing_median_age"]]
        self.transformer.fit(train_data)

    def transform(self, df):
        return self.transformer.transform(df[["housing_median_age"]]).reshape(-1)


class FeatureScaledCleanTotalRooms(BaseHousingTransformingFeature):
    """Feature created by applying box plot to fix outliers then StandardScaler."""

    name = "scaled_total_rooms"

    def fit(self, df):
        self.transformer = StandardScaler()
        train_data = df.query("is_train")[["total_rooms"]]
        self.transformer.fit(train_data)

    def transform(self, df):
        return self.transformer.transform(df[["total_rooms"]]).reshape(-1)


class FeatureScaledCleanTotalBedrooms(BaseHousingTransformingFeature):
    name = "scaled_total_bedrooms"

    def fit(self, df):
        self.transformer = StandardScaler()
        train_data = df.query("is_train")[["total_bedrooms"]]
        self.transformer.fit(train_data)

    def transform(self, df):
        return self.transformer.transform(df[["total_bedrooms"]]).reshape(-1)


class FeatureScaledCleanPopulation(BaseHousingTransformingFeature):
    name = "scaled_population"

    def fit(self, df):
        self.transformer = StandardScaler()
        train_data = df.query("is_train")[["population"]]
        self.transformer.fit(train_data)

    def transform(self, df):
        return self.transformer.transform(df[["population"]]).reshape(-1)


class FeatureScaledCleanHouseholds(BaseHousingTransformingFeature):
    name = "scaled_households"

    def fit(self, df):
        self.transformer = StandardScaler()
        train_data = df.query("is_train")[["households"]]
        self.transformer.fit(train_data)

    def transform(self, df):
        return self.transformer.transform(df[["households"]]).reshape(-1)


class FeatureScaledCleanMedianIncome(BaseHousingTransformingFeature):
    name = "scaled_median_income"

    def fit(self, df):
        self.transformer = StandardScaler()
        train_data = df.query("is_train")[["median_income"]]
        self.transformer.fit(train_data)

    def transform(self, df):
        return self.transformer.transform(df[["median_income"]]).reshape(-1)


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

    def fit(self, df):
        # TODO: explicitly map raw values to coded labels
        self.transformer = LabelEncoder()
        self.transformer.fit(df["ocean_proximity"])

    def transform(self, df):
        return self.transformer.transform(df["ocean_proximity"])


def run():
    feature_config_path = "configs/feature_config.yaml"
    fm = FeatureManager(feature_config_path)
    fm.run_all()


def update_prediction_features():
    feature_config_path = "configs/feature_config.yaml"
    fm = FeatureManager(feature_config_path)
    prediction_feature_names = ["pred_lgbm", "pred_catboost", "pred_xgboost"]
    fm.compute_prediction_features(prediction_feature_names)


if __name__ == "__main__":
    # run()
    update_prediction_features()
