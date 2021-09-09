from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Union

import mlflow
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

from tabml.utils import utils
from tabml.utils.pb_helpers import pb_to_dict

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
        self.feature_names = self.config.data_loader.features_to_model
        self.params: Union[Dict[str, Any], None] = None

    @abstractmethod
    def predict(self, data) -> Iterable:
        """Predicts data inputs.

        data should be the remaining part of dataset without the label column.
        """
        raise NotImplementedError

    def predict_proba(self, data) -> Iterable:
        """Predicts probability of data inputs.

        Only appropriate with binary classification problems, the second value in
        probability (prob of positive) is chosen.
        """
        raise NotImplementedError

    @abstractmethod
    def load_model(self, model_path: str):
        raise NotImplementedError

    @abstractmethod
    def show_feature_importance(self, importance_type: str = "gain") -> None:
        """Displays feature importance after training."""


class BaseLgbmModelWrapper(BaseModelWrapper):
    mlflow_model_type = "lightgbm"

    def __init__(self, config):
        super().__init__(config)
        self.params = pb_to_dict(self.config.model_wrapper.lgbm_params)
        self.model = self.build_model()

    @abstractmethod
    def build_model(self):
        raise NotImplementedError

    def predict(self, data):
        return self.model.predict(data)

    def load_model(self, model_path: str):
        self.model = utils.load_pickle(model_path)

    def show_feature_importance(self, importance_type: str = "gain") -> None:
        importances = self.model.booster_.feature_importance(
            importance_type=importance_type
        )
        assert len(self.feature_names) == len(importances), (
            "feature_names and importances have different lengths "
            f"({len(self.feature_names)} != {len(importances)})"
        )
        data = {self.feature_names[i]: importances[i] for i in range(len(importances))}
        utils.show_feature_importance(data)


class LgbmClassifierModelWrapper(BaseLgbmModelWrapper):
    def build_model(self):
        return LGBMClassifier(**self.params)

    def predict_proba(self, data) -> Iterable:
        return self.model.predict_proba(data)[:, 1]


class LgbmRegressorModelWrapper(BaseLgbmModelWrapper):
    def build_model(self):
        return LGBMRegressor(**self.params)


class BaseXGBoostModelWrapper(BaseModelWrapper):
    mlflow_model_type = "xgboost"

    def __init__(self, config):
        super(BaseXGBoostModelWrapper, self).__init__(config)
        self.params = pb_to_dict(self.config.model_wrapper.xgboost_params)
        self.tree_method = "gpu_hist" if utils.is_gpu_available() else "auto"
        self.model = self.build_model()

    @abstractmethod
    def build_model(self):
        pass

    def predict(self, data):
        return self.model.predict(data)

    def load_model(self, model_path: str):
        self.model = utils.load_pickle(model_path)

    def show_feature_importance(self, importance_type: str = "gain"):
        utils.show_feature_importance(
            self.model.get_booster().get_score(importance_type=importance_type)
        )


class XGBoostRegressorModelWrapper(BaseXGBoostModelWrapper):
    def build_model(self):
        return XGBRegressor(tree_method=self.tree_method, **self.params)


class XGBoostClassifierModelWrapper(BaseXGBoostModelWrapper):
    def build_model(self):
        return XGBClassifier(tree_method=self.tree_method, **self.params)

    def predict_proba(self, data) -> Iterable:
        return self.model.predict_proba(data)[:, 1]


class BaseCatBoostModelWrapper(BaseModelWrapper):
    mlflow_model_type = "catboost"

    def __init__(self, config):
        super(BaseCatBoostModelWrapper, self).__init__(config)
        self.params = pb_to_dict(self.config.model_wrapper.catboost_params)
        self.task_type = "GPU" if utils.is_gpu_available() else "CPU"
        self.model = self.build_model()

    @abstractmethod
    def build_model(self):
        pass

    def predict(self, data):
        return self.model.predict(data)

    def load_model(self, model_path: str):
        self.model = utils.load_pickle(model_path)

    def show_feature_importance(self, importance_type: str = "gain") -> None:
        data = {
            feature: importance
            for (feature, importance) in zip(
                self.feature_names, self.model.get_feature_importance()
            )
        }
        utils.show_feature_importance(data)


class CatBoostClassifierModelWrapper(BaseCatBoostModelWrapper):
    def build_model(self):
        return CatBoostClassifier(**self.params)

    def predict_proba(self, data) -> Iterable:
        return self.model.predict_proba(data)[:, 1]


class CatBoostRegressorModelWrapper(BaseCatBoostModelWrapper):
    def build_model(self):
        return CatBoostRegressor(task_type=self.task_type, **self.params)
