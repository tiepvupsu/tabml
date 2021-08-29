import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

from tabml import datasets
from tabml.feature_manager import BaseFeatureManager, BaseTransformingFeature


class FeatureManager(BaseFeatureManager):
    def __init__(self, pb_config_path):
        super().__init__(pb_config_path)

    def _get_base_transforming_class(self):
        return BaseTitanicTransformingFeature

    def load_raw_data(self):
        df_dict = datasets.download_titanic()
        full_df = pd.concat(
            [df_dict["train"], df_dict["test"]], axis=0, ignore_index=True
        )
        self.raw_data["full"] = full_df

    def set_raw_data(self, raw_data_samples):
        # raw_data_samples is a list of dictionary.
        self.raw_data["full"] = pd.DataFrame(data=raw_data_samples)

    def initialize_dataframe(self):
        self.dataframe = pd.DataFrame()
        self.dataframe["passenger_id"] = self.raw_data["full"]["PassengerId"]
        self.dataframe["sibsp"] = self.raw_data["full"]["SibSp"]
        self.dataframe["parch"] = self.raw_data["full"]["Parch"]
        self.dataframe["fare"] = self.raw_data["full"]["Fare"]
        self.dataframe["age"] = self.raw_data["full"]["Age"]


class BaseTitanicTransformingFeature(BaseTransformingFeature):
    pass


class FeatureIsTrain(BaseTitanicTransformingFeature):
    name = "is_train"

    def transform(self, df):
        np.random.seed(42)
        total_train_samples = 891
        random_array = np.random.rand(len(df))
        # mask test data by 0
        random_array[total_train_samples:] = 0
        validation_size = 0.1
        return np.array(random_array > validation_size, dtype=bool)


class FeatureImputedAge(BaseTitanicTransformingFeature):
    name = "imputed_age"

    def fit(self, df):
        self.transformer = SimpleImputer(strategy="mean")
        train_data = df.query("is_train")[["age"]]
        self.transformer.fit(train_data)

    def transform(self, df):
        return self.transformer.transform(df[["age"]]).reshape(-1)


class FeatureBucketizedAge(BaseTitanicTransformingFeature):
    name = "bucketized_age"
    bins = [10, 18, 30, 40, 65]

    def transform(self, df):
        return np.digitize(df["imputed_age"], self.bins).tolist()


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


class FeatureMinMaxScaledAge(BaseTitanicTransformingFeature):
    name = "min_max_scaled_age"

    def fit(self, df):
        self.transformer = MinMaxScaler()
        train_data = df.query("is_train")[["imputed_age"]]
        self.transformer.fit(train_data)

    def transform(self, df):
        return self.transformer.transform(df[["imputed_age"]]).reshape(-1)


def run():
    pb_config_path = "configs/feature_config.pb"
    fm = FeatureManager(pb_config_path)
    fm.run_all()


if __name__ == "__main__":
    run()
