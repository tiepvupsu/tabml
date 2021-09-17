from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Tuple, Union

import mlflow
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

from tabml.data_loaders import BaseDataLoader
from tabml.utils import utils
from tabml.utils.logger import boosting_logger_eval
from tabml.utils.utils import save_as_pickle

MLFLOW_AUTOLOG = {
    "lightgbm": mlflow.lightgbm.autolog(),
    "xgboost": mlflow.xgboost.autolog(),
    "catboost": None,
}


class BaseModelWrapper(ABC):
    mlflow_model_type = ""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.feature_names = list(self.config.data_loader.features_to_model)
        self.params: Union[Dict[str, Any], None] = None

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


class BaseBoostingModelWrapper(BaseModelWrapper):
    """A common model wrapper for boosting models.

    Boosting models: LightGBM, XGBoost, CatBoost."""

    def __init__(self, config):
        super().__init__(config)
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

    def __init__(self, config):
        super().__init__(config)
        self.params = self.config.model_wrapper.model_params
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
            **self.config.model_wrapper.fit_params,
        }
        return fit_params


class LgbmClassifierModelWrapper(BaseLgbmModelWrapper):
    def build_model(self):
        return LGBMClassifier(**self.params)

    def predict_proba(self, data) -> Iterable:
        return self.model.predict_proba(data)[:, 1]


class LgbmRegressorModelWrapper(BaseLgbmModelWrapper):
    def build_model(self):
        return LGBMRegressor(**self.params)


class BaseXGBoostModelWrapper(BaseBoostingModelWrapper):
    mlflow_model_type = "xgboost"

    def __init__(self, config):
        super(BaseXGBoostModelWrapper, self).__init__(config)
        self.params = self.config.model_wrapper.model_params
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
            **self.config.model_wrapper.fit_params,
        }
        return fit_params


class XGBoostRegressorModelWrapper(BaseXGBoostModelWrapper):
    def build_model(self):
        return XGBRegressor(tree_method=self.tree_method, **self.params)


class XGBoostClassifierModelWrapper(BaseXGBoostModelWrapper):
    def build_model(self):
        return XGBClassifier(tree_method=self.tree_method, **self.params)

    def predict_proba(self, data) -> Iterable:
        return self.model.predict_proba(data)[:, 1]


class BaseCatBoostModelWrapper(BaseBoostingModelWrapper):
    mlflow_model_type = "catboost"

    def __init__(self, config):
        super(BaseCatBoostModelWrapper, self).__init__(config)
        self.params = self.config.model_wrapper.model_params
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
        return CatBoostClassifier(**self.params)

    def predict_proba(self, data) -> Iterable:
        return self.model.predict_proba(data)[:, 1]


class CatBoostRegressorModelWrapper(BaseCatBoostModelWrapper):
    def build_model(self):
        return CatBoostRegressor(task_type=self.task_type, **self.params)
