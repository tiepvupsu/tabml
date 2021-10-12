from tabml.model_wrappers import (
    CatBoostRegressorModelWrapper,
    LgbmRegressorModelWrapper,
    XGBoostRegressorModelWrapper,
)
from tabml.pipelines import BasePipeline


class CustomModelWrapperLog10:
    def predict(self, data):
        # In prediction, the model should return the outputs that are in the same
        # space with the true label.
        return 10 ** self.model.predict(data)


class CustomLgbmRegressorModelWrapperLog10(
    LgbmRegressorModelWrapper, CustomModelWrapperLog10
):
    pass


class CustomXGBoostRegressorModelWrapperLog10(
    XGBoostRegressorModelWrapper, CustomModelWrapperLog10
):
    pass


class CustomCatBoostRegressorModelWrapperLog10(
    CatBoostRegressorModelWrapper, CustomModelWrapperLog10
):
    pass


def train_lgbm():
    path_to_config = "configs/lgbm_config.yaml"
    pipeline = BasePipeline(
        path_to_config, custom_model_wrapper=CustomLgbmRegressorModelWrapperLog10
    )
    pipeline.run()


def train_xgboost():
    path_to_config = "configs/xgboost_config.yaml"
    pipeline = BasePipeline(
        path_to_config, custom_model_wrapper=CustomXGBoostRegressorModelWrapperLog10
    )
    pipeline.run()


def train_catboost():
    path_to_config = "configs/catboost_config.yaml"
    pipeline = BasePipeline(
        path_to_config, custom_model_wrapper=CustomCatBoostRegressorModelWrapperLog10
    )
    pipeline.run()


if __name__ == "__main__":
    train_lgbm()
