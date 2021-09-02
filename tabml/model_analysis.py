import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

import mlflow
import pandas as pd

from tabml.data_loaders import BaseDataLoader
from tabml.metrics import BaseMetric, get_instantiated_metric_dict
from tabml.model_wrappers import BaseModelWrapper
from tabml.utils.logger import logger


class ModelAnalysis:
    """A class performing model analysis on validation dataset on different dimensions.

    For each dimension in features_to_analyze (must be a categorical feature),
    group samples by each possible value then compute the metric scores for the groups.
    Results are then saved in to a csv file with header:
    feature_name, sample_count, metric_0, metric_1, ....
    Rows in each dataframe are sorted by metric_0 column from worst to best (since we
    often want to know how the model performance on the hard groups). Results are saved
    into an output directory.
    Repeat for all dimensions.

    Attributes:
        data_loader:
            A data loader object.
        model_wrapper:
            A model wrapper object.
        features_to_analyze:
            A list of feature names to do the analysis.
        metric_names:
            A list of metric names to be computed.
        output_dir:
            A string of output directory.
        pred_col:
            A string of prediction column (default to "prediction").
        pred_prob_col:
            A string of prediction probability column.
        need_predict:
            A bool value to tell if method self.model_wrapper.predict is needed.
        need_predict_proba:
            A bool value to tell if method self.model_wrapper.predict_proba is needed.
            Some metrics are computed based on probabilities.
    """

    def __init__(
        self,
        data_loader: BaseDataLoader,
        model_wrapper: BaseModelWrapper,
        features_to_analyze: List[str],
        metric_names: List[str],
        output_dir: str,
        label_to_analyze: str = "",
        pred_col: str = "prediction",
        pred_proba_col: str = "prediction_probability",
    ):

        self.data_loader = data_loader
        self.model_wrapper = model_wrapper
        self.features_to_analyze = features_to_analyze
        self.metrics = _get_metrics(metric_names)
        self.output_dir = output_dir
        self.label_to_analyze = label_to_analyze or self.data_loader.label_col
        self.pred_col = pred_col
        self.pred_proba_col = pred_proba_col
        need_pred_proba_list = [metric.need_pred_proba for metric in self.metrics]
        self.need_predict_proba = any(need_pred_proba_list)
        self.need_predict = any(
            not need_pred_proba for need_pred_proba in need_pred_proba_list
        )
        # A bool value to determine when to print the overall scores. This is set to
        # True at initialization, then to False right after printing the first overall
        # scores.
        self._show_overall_flag = True

    def analyze(self):
        self._analyze_one_dataset("train")
        self._analyze_one_dataset("val")

    def _analyze_one_dataset(self, dataset_name: str):
        self._show_overall_flag = True
        dataset = self._get_dataset(dataset_name)
        preds = self._get_predictions(dataset)
        df_with_pred = self._get_df_with_pred(dataset, preds)
        for feature_name in self.features_to_analyze:
            self._analyze_on_feature(df_with_pred, feature_name, dataset_name)

    def _get_dataset(self, dataset_name: str):
        all_features = (
            list(self.features_to_analyze)
            + self.data_loader.features
            + [self.label_to_analyze]
        )
        if dataset_name == "train":
            return self.data_loader.feature_manager.extract_dataframe(
                features_to_select=all_features,
                filters=self.data_loader.config.data_loader.train_filters,
            )
        if dataset_name == "val":
            return self.data_loader.feature_manager.extract_dataframe(
                features_to_select=all_features,
                filters=self.data_loader.config.data_loader.validation_filters,
            )
        raise ValueError(f"dataset_name ({dataset_name}) not in ('train', 'val')")

    def _get_predictions(self, dataset) -> Dict[str, Iterable]:
        res = {}
        if self.need_predict:
            res[self.pred_col] = self.model_wrapper.predict(
                dataset[self.data_loader.features]
            )
        if self.need_predict_proba:
            res[self.pred_proba_col] = self.model_wrapper.predict_proba(
                dataset[self.data_loader.features]
            )
        return res

    def _get_df_with_pred(self, df, preds) -> pd.DataFrame:
        """Appends prediction column to the dataframe."""
        assert (
            self.pred_col not in df.columns
        ), f"{self.pred_col} column already exists in dataset."
        if self.need_predict:
            df[self.pred_col] = preds[self.pred_col]
        if self.need_predict_proba:
            df[self.pred_proba_col] = preds[self.pred_proba_col]
        df.to_csv(self._get_df_pred_csv_path(), index=False)
        return df

    def _analyze_on_feature(
        self, df_with_pred: pd.DataFrame, feature_name: str, dataset_name: str
    ):
        """Analyzes the predictions based on one feature."""
        # get list of metric score dict for each value in feature_name
        list_of_group_dicts = [
            self._get_metric_dict(feature_name, feature_value, group)
            for feature_value, group in df_with_pred.groupby(feature_name)
        ]
        # add overall score
        overall_scores = self._get_metric_dict(feature_name, "OVERALL", df_with_pred)
        list_of_group_dicts.append(overall_scores)
        if self._show_overall_flag:
            self._show_and_log_overall_scores(overall_scores, dataset_name)
            self._show_overall_flag = False

        df_group = pd.DataFrame(list_of_group_dicts).sort_values(
            by=self.metrics[0].score_names[0],
            ascending=self.metrics[0].is_higher_better,
        )
        saved_path = self._get_df_feature_metric_csv_path(dataset_name, feature_name)
        df_group.to_csv(saved_path, index=False)
        logger.info(
            f"Saved model analysis slicing against {feature_name} to {saved_path}"
        )

    def _get_metric_dict(
        self, feature_name: str, feature_value: Any, df_with_pred: pd.DataFrame
    ) -> Dict[str, Union[str, float]]:
        labels = df_with_pred[self.label_to_analyze]

        res = {feature_name: feature_value, "sample_count": len(df_with_pred)}
        for metric in self.metrics:
            if metric.need_pred_proba:
                res.update(
                    metric.compute_scores(labels, df_with_pred[self.pred_proba_col])
                )
            else:
                res.update(metric.compute_scores(labels, df_with_pred[self.pred_col]))
        return res

    def _get_df_pred_csv_path(self):
        return os.path.join(self.output_dir, "prediction.csv")

    def _get_df_feature_metric_csv_path(self, dataset_name: str, col: str):
        dirname = Path(self.output_dir) / dataset_name
        if not dirname.exists():
            dirname.mkdir()

        return dirname / f"{col}.csv"

    def _show_and_log_overall_scores(
        self, overall_scores: Dict[str, Any], dataset_name: str
    ) -> None:
        logger.info("=" * 20 + f" OVERALL SCORES on {dataset_name} dataset " + "=" * 20)
        logger.info("{:<20}: {}".format("Num samples", overall_scores["sample_count"]))
        for key, val in overall_scores.items():
            if val == "OVERALL" or key == "sample_count":
                continue
            logger.info("{:<20}: {}".format(key, val))
            mlflow.log_metrics({key: val})


def _get_metrics(metric_names: List[str]) -> List[BaseMetric]:
    metric_by_name = get_instantiated_metric_dict()
    return [metric_by_name[metric_name] for metric_name in metric_names]
