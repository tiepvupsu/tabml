from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from tabml.model_wrappers import NewBaseModelWrapper
from tabml.pipelines import BasePipeline


def run(path_to_config: str):
    pipeline = BasePipeline(path_to_config)
    pipeline.run()


def train_lgbm():
    path_to_config = "configs/lgbm_config.pb"
    lgbm_model_wrapper = NewBaseModelWrapper(
        LGBMClassifier,
        model_params={},
        fit_params={
            "categorical_feature": ["coded_pclass", "coded_title", "coded_sex"]
        },
        mlflow_model_type="lightgbm",
    )
    pipeline = BasePipeline(path_to_config, custom_model_wrapper=lgbm_model_wrapper)
    pipeline.run()


def train_xgboost():
    path_to_config = "configs/xgboost_config.pb"
    xgboost_model_wrapper = NewBaseModelWrapper(
        XGBClassifier,
        model_params={
            "n_estimators": 200,
            "objective": "reg:squarederror",
            "eval_metric": "auc",
        },
        mlflow_model_type="xgboost",
    )
    pipeline = BasePipeline(path_to_config, custom_model_wrapper=xgboost_model_wrapper)
    pipeline.run()


def train_catboost():
    path_to_config = "configs/catboost_config.pb"
    catboost_model_wrapper = NewBaseModelWrapper(
        CatBoostClassifier,
        model_params={
            "n_estimators": 100,
            "loss_function": "CrossEntropy",
            "eval_metric": "AUC",
        },
        mlflow_model_type="catboost",
    )
    pipeline = BasePipeline(path_to_config, custom_model_wrapper=catboost_model_wrapper)
    pipeline.run()


def train_randomforest():
    path_to_config = "configs/catboost_config.pb"
    catboost_model_wrapper = NewBaseModelWrapper(
        RandomForestClassifier, model_params={}, mlflow_model_type="sklearn"
    )
    pipeline = BasePipeline(path_to_config, custom_model_wrapper=catboost_model_wrapper)
    pipeline.run()


if __name__ == "__main__":
    # train_lgbm()
    # train_xgboost()
    # train_catboost()
    train_randomforest()
