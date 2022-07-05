from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, Tuple, Union

import mlflow
import numpy as np
import shap
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

from tabml.config_helpers import parse_pipeline_config
from tabml.data_loaders import BaseDataLoader
from tabml.experiment_manager import ExperimentManger
from tabml.schemas import pipeline_config
from tabml.utils import factory, utils
from tabml.utils.logger import boosting_logger_eval, logger
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
        self.save_model_name = None
        # Parameters for model instantiating
        self.model_params = params.model_params
        # Parameters for model training
        self.fit_params = params.fit_params

    def fit(self, data_loader: BaseDataLoader, model_dir: str):
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
    def load_model(self, model_path: str):
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
        pass

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
        pass

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


def initialize_model_wrapper(
    params: pipeline_config.ModelWrapper, model_path: Union[str, None] = None
):
    """Initializes model wrapper from params."""
    model_wrapper_cls = factory.create(params.cls_name)
    if not issubclass(model_wrapper_cls, BaseModelWrapper):
        raise ValueError(f"{model_wrapper_cls} is not a subclass of BaseModelWrapper")
    _model_wrapper = model_wrapper_cls(params)
    if model_path:
        _model_wrapper.load_model(model_path)
    return _model_wrapper


def load_or_train_model(model_path, pipeline_config_path) -> BaseModelWrapper:
    """Loads or trains a model, returns a model wrapper."""
    if not (model_path or pipeline_config_path):
        raise ValueError(
            "At least one of model_path and pipeline_config_path must be not None."
        )

    if not pipeline_config_path:
        pipeline_config_path = ExperimentManger.get_config_path_from_model_path(
            model_path
        )

    if not model_path:
        try:
            logger.info(
                f"Searching for the last run dir with {pipeline_config_path} config."
            )
            run_dir = ExperimentManger(pipeline_config_path).get_most_recent_run_dir()
            # TODO: create a function/method in experiment_manager to find model_path
            # in run_dir.
            model_path = str(Path(run_dir) / "model_0")
        except IOError:
            import tabml.pipelines

            pipeline = tabml.pipelines.BasePipeline(pipeline_config_path)
            pipeline.run()
            return pipeline.model_wrapper

    config = parse_pipeline_config(pipeline_config_path)
    return initialize_model_wrapper(config.model_wrapper, model_path)


if __name__ == "__main__":
    # Show the subclasses of BaseModelWrapper
    write_model_wrapper_subclasses_to_file()
