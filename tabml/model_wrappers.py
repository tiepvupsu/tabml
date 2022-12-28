from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, Tuple, Union

import numpy as np
import shap
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

from tabml.data_loaders import BaseDataLoader
from tabml.schemas import pipeline_config
from tabml.schemas.pipeline_config import ModelBundle
from tabml.utils import factory, utils
from tabml.utils.logger import boosting_logger_eval
from tabml.utils.utils import load_pickle


class BaseModelWrapper(ABC):
    mlflow_model_type = ""

    def __init__(self, params: pipeline_config.ModelWrapper, model=None):
        self.model = model
        self.save_model_name = "model_0"
        # Parameters for model instantiating
        self.model_params = params.model_params
        # Parameters for model training
        self.fit_params = params.fit_params

    def fit(self, data_loader: BaseDataLoader, model_dir: Union[str, Path]):
        pass

    @abstractmethod
    def predict(self, data) -> Iterable:
        """Predicts data inputs."""
        pass

    def predict_proba(self, data) -> Iterable:
        """Predicts probability of data inputs.

        Only applied to binary classification problems, the second value in
        probability (prob of positive) is chosen.
        """
        raise NotImplementedError

    @abstractmethod
    def load_model(self, model_path: Union[str, Path]):
        pass

    def get_feature_importance(self, input_data) -> Dict[str, float]:
        """Computes feature importance for each feature based on an input data.

        Most of models are supported by SHAP (https://github.com/slundberg/shap). For
        unsupported models, please override this method by a workable solution.
        """
        explainer = shap.Explainer(self.model.predict, input_data)
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
        self.model = factory.create(params.sklearn_cls)(**self.model_params)

    def fit(self, data_loader: BaseDataLoader, model_dir: Union[str, Path]):
        assert (
            data_loader.label_col is not None
        ), "data_loader.label_col must be declared in BaseDataLoader subclasses."
        train_feature, train_label = data_loader.get_train_data_and_label()

        self.model.fit(X=train_feature, y=train_label, **self.fit_params)

    def load_model(self, model_path: Union[str, Path]):
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
        self.build_model()

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def _get_fit_params(self, train_data: Tuple, val_data: Tuple) -> Dict:
        pass

    def fit(self, data_loader: BaseDataLoader, model_dir: Union[str, Path]):
        assert (
            data_loader.label_col is not None
        ), "data_loader.label_col must be declared in BaseDataLoader subclasses."
        train_feature, train_label = data_loader.get_train_data_and_label()
        val_data = data_loader.get_val_data_and_label()

        fit_params = self._get_fit_params((train_feature, train_label), val_data)

        self.model.fit(X=train_feature, y=train_label, **fit_params)


class BaseLgbmModelWrapper(BaseBoostingModelWrapper):
    mlflow_model_type = "lightgbm"

    def __init__(self, params=pipeline_config.ModelWrapper()):
        super().__init__(params)

    def predict(self, data):
        return self.model.predict(data)

    def load_model(self, model_path: Union[str, Path]):
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
        self.model = LGBMClassifier(**self.model_params)

    def predict_proba(self, data) -> Iterable:
        return self.model.predict_proba(data)[:, 1]


class LgbmRegressorModelWrapper(BaseLgbmModelWrapper):
    def build_model(self):
        self.model = LGBMRegressor(**self.model_params)


class BaseXGBoostModelWrapper(BaseBoostingModelWrapper):
    mlflow_model_type = "xgboost"
    tree_method = "gpu_hist" if utils.is_gpu_available() else "auto"

    def __init__(self, params=pipeline_config.ModelWrapper()):
        super().__init__(params)

    def predict(self, data):
        return self.model.predict(data)

    def load_model(self, model_path: Union[str, Path]):
        self.model = utils.load_pickle(model_path)

    def _get_fit_params(self, train_data, val_data):
        # TODO: add callback to display train and validation metrics
        fit_params = {
            "eval_set": [train_data, val_data],
            **self.fit_params,
        }
        return fit_params


class XGBoostRegressorModelWrapper(BaseXGBoostModelWrapper):
    def build_model(self):
        self.model = XGBRegressor(tree_method=self.tree_method, **self.model_params)


class XGBoostClassifierModelWrapper(BaseXGBoostModelWrapper):
    def build_model(self):
        self.model = XGBClassifier(tree_method=self.tree_method, **self.model_params)

    def predict_proba(self, data) -> Iterable:
        return self.model.predict_proba(data)[:, 1]


class BaseExponentialWeightLgbmModelWrapper(BaseLgbmModelWrapper):
    def __init__(self, params=pipeline_config.ModelWrapper()):
        super().__init__(params)
        self._get_sample_weight_params(params)
        self.sample_weights = None

    def _get_sample_weight_params(self, params):
        allowed_params = ["scale", "decay", "num_same_weight_samples"]
        default_scale = 1
        default_decay = 40
        default_num_same_weight_sample = 1
        weight_params = params.weight_params
        invalid_params = [
            param for param in weight_params.keys() if param not in allowed_params
        ]
        if invalid_params:
            raise ValueError(
                f"weight_params only allows {allowed_params}. "
                f"Found {invalid_params}."
            )
        self.scale = weight_params.get("scale", default_scale)
        self.decay = weight_params.get("decay", default_decay)
        self.num_same_weight_samples = weight_params.get(
            "num_same_weight_samples", default_num_same_weight_sample
        )

    def _compute_exponential_weight(self, data_loader: BaseDataLoader):
        feature_to_create_weights = data_loader.feature_to_create_weights
        if feature_to_create_weights is None:
            raise ValueError(
                "Please define feature_to_create_weights in DataLoader config."
            )
        feature_to_create_weights_values = (
            data_loader.feature_manager.extract_dataframe(
                features_to_select=[feature_to_create_weights],
                filters=data_loader.train_filters,
            )[feature_to_create_weights]
        )
        max_feature_value = feature_to_create_weights_values.max()
        weight = max_feature_value - feature_to_create_weights_values
        weight = weight / self.num_same_weight_samples
        weight = weight.astype("int")
        weight = np.exp(-weight / self.decay)
        weight = self.scale * weight
        return weight

    def fit(self, data_loader: BaseDataLoader, model_dir: Union[str, Path]):
        self.sample_weights = self._compute_exponential_weight(data_loader)
        self.fit_params["sample_weight"] = self.sample_weights
        super(BaseExponentialWeightLgbmModelWrapper, self).fit(data_loader, model_dir)


class ExponentialWeightLgbmClassifierModelWrapper(
    BaseExponentialWeightLgbmModelWrapper, LgbmClassifierModelWrapper
):
    def __init__(self, params=pipeline_config.ModelWrapper()):
        super().__init__(params)


class ExponentialWeightLgbmRegressorModelWrapper(
    BaseExponentialWeightLgbmModelWrapper, LgbmRegressorModelWrapper
):
    def __init__(self, params=pipeline_config.ModelWrapper()):
        super().__init__(params)


class BaseCatBoostModelWrapper(BaseBoostingModelWrapper):
    mlflow_model_type = "catboost"
    task_type = "GPU" if utils.is_gpu_available() else "CPU"

    def __init__(self, params=pipeline_config.ModelWrapper()):
        super().__init__(params)

    def predict(self, data):
        return self.model.predict(data)

    def load_model(self, model_path: Union[str, Path]):
        self.model = utils.load_pickle(model_path)

    def _get_fit_params(self, train_data, val_data):
        fit_params = {"eval_set": [val_data]}

        return fit_params


class CatBoostClassifierModelWrapper(BaseCatBoostModelWrapper):
    def build_model(self):
        self.model = CatBoostClassifier(**self.model_params)

    def predict_proba(self, data) -> Iterable:
        return self.model.predict_proba(data)[:, 1]


class CatBoostRegressorModelWrapper(BaseCatBoostModelWrapper):
    def build_model(self):
        self.model = CatBoostRegressor(task_type=self.task_type, **self.model_params)


def write_model_wrapper_subclasses_to_file(
    base_cls=BaseModelWrapper, md_path="model_wrapper_map.md"
):
    lines = ["# Inheritance map\n", "\n"]
    level = 0
    stack = [(base_cls, level)]
    while stack:
        node, level = stack.pop()
        lines.append("    " * level + f"- {node}\n")
        stack.extend(
            (child, level + 1) for child in node.__subclasses__()
        )  # type: ignore

    with open(md_path, "w") as f:
        f.writelines(lines)


def initialize_model_wrapper(model_bundle: Union[str, Path, ModelBundle]):
    _model_bundle = (
        model_bundle
        if isinstance(model_bundle, ModelBundle)
        else load_pickle(model_bundle)
    )
    model_wrapper_params = _model_bundle.pipeline_config.model_wrapper
    model_wrapper_cls = factory.create(model_wrapper_params.cls_name)
    if not issubclass(model_wrapper_cls, BaseModelWrapper):
        raise ValueError(f"{model_wrapper_cls} is not a subclass of BaseModelWrapper")
    _model_wrapper = model_wrapper_cls(model_wrapper_params)
    if _model_bundle.model:
        _model_wrapper.model = _model_bundle.model
    return _model_wrapper


if __name__ == "__main__":
    # Show the subclasses of BaseModelWrapper
    write_model_wrapper_subclasses_to_file()
