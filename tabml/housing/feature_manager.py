from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
            ["median_house_value", "housing_median_age", "total_rooms"]
        ]


class BaseHousingTransformingFeature(BaseTransformingFeature):
    pass


class FeatureAge(BaseHousingTransformingFeature):
    name = "is_train"

    def transform(self, df):
        random_array = np.random.rand(len(df))
        validation_size = 0.1
        return random_array > validation_size


class FeatureHousingMedianAge(BaseHousingTransformingFeature):
    name = "scaled_housing_median_age"

    def transform(self, df):
        # fit StandardScaler() on train data, transform whole data
        # NOTE: StandardScaler requires 2D inputs
        train_data = df.query("is_train")[["housing_median_age"]]
        scaler = StandardScaler()
        scaler.fit(train_data)
        # return a 1d array for pandas series
        return scaler.transform(df[["housing_median_age"]]).reshape(-1)


class FeatureScaledCleanTotalRooms(BaseHousingTransformingFeature):
    """Feature created by applying box plot to fix outliers then StandardScaler."""

    name = "scaled_clean_total_rooms"

    def transform(self, df):
        train_data = df.query("is_train")[["total_rooms"]]
        all_data = df[["total_rooms"]]
        transformer = Pipeline(
            [
                ("clip_outlier", data_processing.BoxplotOutlierClipper()),
                ("std_scaler", StandardScaler()),
            ]
        )
        transformer.fit(train_data)
        return transformer.transform(all_data).reshape(-1)


def run():
    pb_config_path = (
        Path(get_git_repo_dir()) / "tabml/housing/configs/feature_config.pb"
    )
    fm = FeatureManager(pb_config_path)
    fm.run_all()


if __name__ == "__main__":
    run()
