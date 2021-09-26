from abc import ABC, abstractmethod
from typing import Dict, Iterable, Tuple

import mlflow
import numpy as np
import shap
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

from tabml.data_loaders import BaseDataLoader
from tabml.schemas import pipeline_config
from tabml.utils import factory, utils
from tabml.utils.logger import boosting_logger_eval
from tabml.utils.utils import save_as_pickle

MLFLOW_AUTOLOG = {
    "sklearn": mlflow.sklearn.autolog(),
    "lightgbm": mlflow.lightgbm.autolog(),
    "xgboost": mlflow.xgboost.autolog(),
    "catboost": None,
}


class BaseModelWrapper(ABC):
    mlflow_model_type = ""

    def __init__(self, params=pipeline_config.ModelWrapper()):
        self.model = None
        # Parameters for model instantiating
        self.model_params = params.model_params
        # Parameters for model training
        self.fit_params = params.fit_params

    def fit(self, data_loader: BaseDataLoader, model_dir: str):
        pass

    @abstractmethod
    def predict(self, data) -> Iterable:
        """Predicts data inputs."""
        raise NotImplementedError

    def predict_proba(self, data) -> Iterable:
        """Predicts probability of data inputs.

        Only applied to binary classification problems, the second value in
        probability (prob of positive) is chosen.
        """
        raise NotImplementedError

    @abstractmethod
    def load_model(self, model_path: str):
        raise NotImplementedError

    def get_feature_importance(self, input_data) -> Dict[str, float]:
        """Computes feature importance for each feature based on an input data.

        Most of models are supported by SHAP (https://github.com/slundberg/shap). For
        unsupported models, please override this method by a workable solution.
        """
        explainer = shap.Explainer(self.model)
        shap_values = explainer(input_data)

        def _get_shap_values_one_sample(shap_values, index: int):
            # For LightGBM and XGBoost, shap_values[index].values is a 2d array
            # representing logit of two classes. They are basically negative of each
            # other, we only need one.
            # Related issue https://github.com/slundberg/shap/issues/526.
            if len(shap_values[index].values.shape) == 2:  # binary classification
                return shap_values[index].values[:, 0]
            assert len(shap_values[index].values.shape) == 1, len(
                shap_values[index].values
            )
            return shap_values[index].values

        feature_importances = np.mean(
            [
                np.abs(_get_shap_values_one_sample(shap_values, i))
                for i in range(len(shap_values))
            ],
            axis=0,
        ).tolist()

        feature_names = input_data.columns.tolist()
        feature_importance_dict = dict(zip(feature_names, feature_importances))
        return feature_importance_dict


class BaseSklearnModelWrapper(BaseModelWrapper):
    """A common model wrapper for scklearn-like models."""

    mlflow_model_type = "sklearn"

    def __init__(self, params=pipeline_config.ModelWrapper()):
        super().__init__(params)
        self.save_model_name = "model_0"
        self.model = factory.create(params.model_cls)(**self.model_params)

    def fit(self, data_loader: BaseDataLoader, model_dir: str):
        assert (
            data_loader.label_col is not None
        ), "data_loader.label_col must be declared in BaseDataLoader subclasses."
        train_feature, train_label = data_loader.get_train_data_and_label()

        self.model.fit(X=train_feature, y=train_label, **self.fit_params)
        save_as_pickle(self.model, model_dir, self.save_model_name)

    def load_model(self, model_path: str):
        self.model = utils.load_pickle(model_path)

    def predict(self, data):
        return self.model.predict(data)

    def predict_proba(self, data) -> Iterable:
        return self.model.predict_proba(data)[:, 1]


class BaseBoostingModelWrapper(BaseModelWrapper):
    """A common model wrapper for boosting models.

    Boosting models: LightGBM, XGBoost, CatBoost."""

    def __init__(self, params=pipeline_config.ModelWrapper()):
        super().__init__(params)
        self.save_model_name = "model_0"

    @abstractmethod
    def _get_fit_params(self, train_data: Tuple, val_data: Tuple) -> Dict:
        raise NotImplementedError

    def fit(self, data_loader: BaseDataLoader, model_dir: str):
        assert (
            data_loader.label_col is not None
        ), "data_loader.label_col must be declared in BaseDataLoader subclasses."
        train_feature, train_label = data_loader.get_train_data_and_label()
        val_data = data_loader.get_val_data_and_label()

        fit_params = self._get_fit_params((train_feature, train_label), val_data)

        # breakpoint()
        self.model.fit(X=train_feature, y=train_label, **fit_params)
        save_as_pickle(self.model, model_dir, self.save_model_name)


class BaseLgbmModelWrapper(BaseBoostingModelWrapper):
    mlflow_model_type = "lightgbm"

    def __init__(self, params=pipeline_config.ModelWrapper()):
        super().__init__(params)
        self.model = self.build_model()

    @abstractmethod
    def build_model(self):
        raise NotImplementedError

    def predict(self, data):
        return self.model.predict(data)

    def load_model(self, model_path: str):
        self.model = utils.load_pickle(model_path)

    def _get_fit_params(self, train_data: Tuple, val_data: Tuple) -> Dict:
        fit_params = {
            "eval_set": [train_data, val_data],
            "eval_names": ["train", "val"],
            "callbacks": [boosting_logger_eval(model="lgbm")],
            **self.fit_params,
        }
        return fit_params


class LgbmClassifierModelWrapper(BaseLgbmModelWrapper):
    def build_model(self):
        return LGBMClassifier(**self.model_params)

    def predict_proba(self, data) -> Iterable:
        return self.model.predict_proba(data)[:, 1]


class LgbmRegressorModelWrapper(BaseLgbmModelWrapper):
    def build_model(self):
        return LGBMRegressor(**self.model_params)


class BaseXGBoostModelWrapper(BaseBoostingModelWrapper):
    mlflow_model_type = "xgboost"

    def __init__(self, params=pipeline_config.ModelWrapper):
        super().__init__(params)
        self.tree_method = "gpu_hist" if utils.is_gpu_available() else "auto"
        self.model = self.build_model()

    @abstractmethod
    def build_model(self):
        pass

    def predict(self, data):
        return self.model.predict(data)

    def load_model(self, model_path: str):
        self.model = utils.load_pickle(model_path)

    def _get_fit_params(self, train_data, val_data):
        fit_params = {
            "eval_set": [train_data, val_data],
            "callbacks": [boosting_logger_eval(model="xgboost")],
            **self.fit_params,
        }
        return fit_params


class XGBoostRegressorModelWrapper(BaseXGBoostModelWrapper):
    def build_model(self):
        return XGBRegressor(tree_method=self.tree_method, **self.model_params)


class XGBoostClassifierModelWrapper(BaseXGBoostModelWrapper):
    def build_model(self):
        return XGBClassifier(tree_method=self.tree_method, **self.model_params)

    def predict_proba(self, data) -> Iterable:
        return self.model.predict_proba(data)[:, 1]


class BaseCatBoostModelWrapper(BaseBoostingModelWrapper):
    mlflow_model_type = "catboost"

    def __init__(self, params=pipeline_config.ModelWrapper()):
        super().__init__(params)
        self.task_type = "GPU" if utils.is_gpu_available() else "CPU"
        self.model = self.build_model()

    @abstractmethod
    def build_model(self):
        pass

    def predict(self, data):
        return self.model.predict(data)

    def load_model(self, model_path: str):
        self.model = utils.load_pickle(model_path)

    def _get_fit_params(self, train_data, val_data):
        fit_params = {"eval_set": [val_data]}

        return fit_params


class CatBoostClassifierModelWrapper(BaseCatBoostModelWrapper):
    def build_model(self):
        return CatBoostClassifier(**self.model_params)

    def predict_proba(self, data) -> Iterable:
        return self.model.predict_proba(data)[:, 1]


class CatBoostRegressorModelWrapper(BaseCatBoostModelWrapper):
    def build_model(self):
        return CatBoostRegressor(task_type=self.task_type, **self.model_params)


def write_model_wrapper_subclasses_to_file(
    base_cls=BaseModelWrapper, md_path="model_wrapper_map.md"
):
    lines = ["# Inheritance map\n", "\n"]
    level = 0
    stack = [(base_cls, level)]
    while stack:
        node, level = stack.pop()
        lines.append("    " * level + f"- {node}\n")
        stack.extend((child, level + 1) for child in node.__subclasses__())

    with open(md_path, "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    # Show the subclasses of BaseModelWrapper
    write_model_wrapper_subclasses_to_file()
