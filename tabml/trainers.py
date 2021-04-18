from abc import ABC, abstractmethod
from typing import Dict, Tuple

import pandas as pd

from tabml import data_loaders, model_wrappers
from tabml.utils.logger import boosting_logger_eval
from tabml.utils.pb_helpers import pb_to_dict
from tabml.utils.utils import save_as_pickle


class BaseTrainer(ABC):
    def __init__(
        self,
        model_wrapper: model_wrappers.BaseModelWrapper,
        data_loader: data_loaders.BaseDataLoader,
        config,
    ):
        self.model_wrapper = model_wrapper
        self.data_loader = data_loader
        self.config = config
        self.train_full: bool = self.config.trainer.train_full

    @abstractmethod
    def train(self, model_dir: str):
        raise NotImplementedError


class BaseBoostingTrainer(BaseTrainer):
    def __init__(
        self,
        model_wrapper: model_wrappers.BaseModelWrapper,
        data_loader: data_loaders.BaseDataLoader,
        config,
    ):
        super().__init__(model_wrapper, data_loader, config)
        # save as model_0 to make it similar to keras saved models
        self.save_model_name = "model_0"

    @abstractmethod
    def _get_fit_params(
        self,
        train_data: Tuple[pd.DataFrame, pd.Series],
        val_data: Tuple[pd.DataFrame, pd.Series],
    ) -> Dict:
        raise NotImplementedError

    def train(self, model_dir: str) -> None:
        assert (
            self.data_loader.label_col is not None
        ), "self.data_loader.label_col must be declared in BaseDataLoader subclasses."
        train_feature, train_label = self.data_loader.get_train_data_and_label()
        val_data = self.data_loader.get_val_data_and_label()

        self.model_wrapper.feature_names = train_feature.columns

        fit_params = self._get_fit_params((train_feature, train_label), val_data)

        self.model_wrapper.model.fit(X=train_feature, y=train_label, **fit_params)
        self.model_wrapper.show_feature_importance()
        save_as_pickle(self.model_wrapper.model, model_dir, self.save_model_name)


class LgbmTrainer(BaseBoostingTrainer):
    def _get_fit_params(self, train_data, val_data):
        fit_params = {
            "eval_set": [train_data, val_data],
            "eval_names": ["train", "val"],
            "callbacks": [boosting_logger_eval(model="lgbm")],
            **pb_to_dict(self.config.trainer.lgbm_params),
        }
        return fit_params
