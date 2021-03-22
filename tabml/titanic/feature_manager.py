from pathlib import Path

import numpy as np
import pandas as pd

from tabml.feature_manager import BaseFeatureManager, BaseTransformingFeature
from tabml.utils.git import get_git_repo_dir


class FeatureManager(BaseFeatureManager):
    def __init__(self, pb_config_path):
        super().__init__(pb_config_path)

    def _get_base_transforming_class(self):
        return BaseTitanicTransformingFeature

    def load_raw_data(self):
        raw_data_dir = Path(self.raw_data_dir)
        train_df = pd.read_csv(raw_data_dir / "train.csv")
        test_df = pd.read_csv(raw_data_dir / "test.csv")
        full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
        self.raw_data["full"] = full_df

    def initialize_dataframe(self):
        self.dataframe = pd.DataFrame()
        self.dataframe["passenger_id"] = self.raw_data["full"]["PassengerId"]
        self.dataframe["sibsp"] = self.raw_data["full"]["SibSp"]
        self.dataframe["parch"] = self.raw_data["full"]["Parch"]
        self.dataframe["fare"] = self.raw_data["full"]["Fare"]


class BaseTitanicTransformingFeature(BaseTransformingFeature):
    pass


class FeatureAge(BaseTitanicTransformingFeature):
    name = "age"

    def transform(self, df):
        mean_age = self.raw_data["full"]["Age"].mean()
        return self.raw_data["full"]["Age"].fillna(mean_age)


class FeatureBucketizedAge(BaseTitanicTransformingFeature):
    name = "bucketized_age"
    bins = [10, 18, 30, 40, 65]

    def transform(self, df):
        return np.digitize(df["age"], self.bins).tolist()


class FeatureSurvived(BaseTitanicTransformingFeature):
    name = "survived"

    def transform(self, df):
        return self.raw_data["full"]["Survived"]


class FeatureSex(BaseTitanicTransformingFeature):
    name = "sex"

    def transform(self, df):
        return self.raw_data["full"]["Sex"]


class FeatureCodedSex(BaseTitanicTransformingFeature):
    name = "coded_sex"

    def transform(self, df):
        return df["sex"].map({"male": 0, "female": 1})


class FeaturePclass(BaseTitanicTransformingFeature):
    name = "pclass"

    def transform(self, df):
        return self.raw_data["full"]["Pclass"]


class FeatureCodedPclass(BaseTitanicTransformingFeature):
    name = "coded_pclass"

    def transform(self, df):
        return df["pclass"].map({1: 1, 2: 2, 3: 0})


class FeatureEmbarked(BaseTitanicTransformingFeature):
    name = "embarked"

    def transform(self, df):
        embarked = self.raw_data["full"]["Embarked"]
        most_frequent = embarked.value_counts().index[0]
        return embarked.fillna(most_frequent)


class FeatureCodedEmbarked(BaseTitanicTransformingFeature):
    name = "coded_embarked"

    def transform(self, df):
        return df["embarked"].map({"C": 0, "Q": 1, "S": 2, "Unknown": 3})


class FeatureTitle(BaseTitanicTransformingFeature):
    name = "title"

    def transform(self, df):
        title = self.raw_data["full"]["Name"].str.extract(
            r" ([A-Za-z]+)\.", expand=False
        )
        title = title.replace(
            [
                "Dr",
                "Rev",
                "Col",
                "Major",
                "Countess",
                "Sir",
                "Jonkheer",
                "Lady",
                "Capt",
                "Don",
                "Dona",
            ],
            "Others",
        )
        title = title.replace("Ms", "Miss")
        title = title.replace("Mme", "Mrs")
        title = title.replace("Mlle", "Miss")
        return title


class FeatureCodedTitle(BaseTitanicTransformingFeature):
    name = "coded_title"

    def transform(self, df):
        return df["title"].map(
            {"Master": 0, "Miss": 1, "Mr": 2, "Mrs": 3, "Others": 4, "Unknown": 5}
        )


def run():
    pb_config_path = (
        Path(get_git_repo_dir()) / "tabml/titanic/configs/feature_config.pb"
    )
    fm = FeatureManager(pb_config_path)
    fm.run_all()


if __name__ == "__main__":
    run()
