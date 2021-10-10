from tabml.pipelines import BasePipeline

from . import model_wrappers


def train_lgbm():
    path_to_config = "configs/lgbm_config.yaml"
    pipeline = BasePipeline(path_to_config)
    pipeline.run()


def train_xgboost():
    path_to_config = "configs/xgboost_config.yaml"
    pipeline = BasePipeline(
        path_to_config,
        custom_model_wrapper=model_wrappers.CustomXGBoostRegressorModelWrapperLog10,
    )
    pipeline.run()


def train_catboost():
    path_to_config = "configs/catboost_config.yaml"
    pipeline = BasePipeline(
        path_to_config,
        custom_model_wrapper=model_wrappers.CustomCatBoostRegressorModelWrapperLog10,
    )
    pipeline.run()


if __name__ == "__main__":
    train_lgbm()
