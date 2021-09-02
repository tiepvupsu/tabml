from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Union

import mlflow
from lightgbm import LGBMClassifier, LGBMRegressor

from tabml.utils import utils
from tabml.utils.pb_helpers import pb_to_dict

MLFLOW_AUTOLOG = {"lightgbm": mlflow.lightgbm.autolog()}


class BaseModelWrapper(ABC):
    mlflow_model_type = ""

    def __init__(self, config):
        self.config = config
        self.model = None
        self._feature_names = None
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

    @property
    def feature_names(self):
        assert (
            self._feature_names is not None
        ), "self._feature_names has not been set. Please set it first."
        return self._feature_names

    @feature_names.setter
    def feature_names(self, names):
        """Sets feature names during training. This should be set only once."""
        assert (
            self._feature_names is None
        ), f"feature_names has been set before, feature_names = {self.feature_names}"
        self._feature_names = names


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
