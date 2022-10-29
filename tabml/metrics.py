import abc
from typing import Dict, List, Sequence, Union

import numpy as np
from sklearn import metrics as sk_metrics


class BaseMetric(abc.ABC):
    """Base class for metrics.

    Usually, one metric returns one score like in the case of accuracy, rmse, mse, etc.
    However, in some cases, metrics might contain several values as precision, recall,
    f1. In these cases, method compute_scores will return a list of scores.

    Attributes:
        name:
            A string of class attribute, unique for each metric.
        is_higer_better:
            A bool value to tell if the higher the metric, the better the performance.
            The metric here is the first score in the output of compute_scores function.
            Note that, for some metrics like RMSE, lower numbers are better.
        need_pred_proba:
            A bool value to decide predict or predict_proba in model will be used.
    """

    name: str = ""
    score_names = [""]
    is_higher_better: Union[bool, None] = None
    need_pred_proba: bool = False

    def compute_scores(self, labels: Sequence, preds: Sequence) -> Dict[str, float]:
        if self.is_higher_better is None:
            raise ValueError("Subclasses must define is_higher_better.")
        if len(labels) != len(preds):
            raise ValueError(
                f"labels (len = {len(labels)}) and preds (len = {len(preds)}) "
                "must have the same length"
            )
        scores = self._compute_scores(labels, preds)
        if len(self.score_names) != len(scores):
            raise ValueError(
                f"self.score_names ({self.score_names}) and scores ({scores}) "
                "must have the same length."
            )
        return dict(zip(self.score_names, scores))

    def _compute_scores(self, labels: Sequence, preds: Sequence) -> List[float]:
        raise NotImplementedError


class MAE(BaseMetric):
    """Mean Absolute Error."""

    name = "mae"
    score_names = ["mae"]
    is_higher_better = False

    def _compute_scores(self, labels: Sequence, preds: Sequence) -> List[float]:
        return [sk_metrics.mean_absolute_error(labels, preds)]


class RMSE(BaseMetric):
    """Root Mean Square Error."""

    name = "rmse"
    score_names = ["rmse"]
    is_higher_better = False

    def _compute_scores(self, labels: Sequence, preds: Sequence) -> List[float]:
        return [sk_metrics.mean_squared_error(labels, preds) ** 0.5]


class AccuracyScore(BaseMetric):
    """Accuracy for classification."""

    name = "accuracy_score"
    score_names = ["accuracy_score"]
    is_higher_better = True

    def _compute_scores(self, labels: Sequence, preds: Sequence) -> List[float]:
        return [sk_metrics.accuracy_score(labels, preds)]


class RocAreaUnderTheCurve(BaseMetric):
    """Area ROC under the curve for binary classification."""

    name = "roc_auc"
    score_names = ["roc_auc"]
    is_higher_better = True
    need_pred_proba = True

    def _compute_scores(self, labels: Sequence, preds: Sequence) -> List[float]:
        # In some small groups of samples, labels might be all 0 or 1, which might cause
        # ValueError when computing ROC AUC.
        try:
            res = sk_metrics.roc_auc_score(labels, preds)
        except ValueError as error_message:
            if (
                "Only one class present in y_true. "
                "ROC AUC score is not defined in that case." in repr(error_message)
            ):
                res = np.NaN
            else:
                raise ValueError(error_message)
        return [res]


class F1Score(BaseMetric):
    """F1 score."""

    name = "f1"
    score_names = ["f1", "precision", "recall"]
    is_higher_better = True
    need_pred_proba = False

    def _compute_scores(self, labels: Sequence, preds: Sequence) -> List[float]:
        _check_binary_list(labels)
        _check_binary_list(preds)
        labels_ = np.array(labels)
        preds_ = np.array(preds)
        eps = 1e-8
        tps = np.sum(np.logical_and(preds_ == 1, labels_ == 1))
        fps = np.sum(np.logical_and(preds_ == 1, labels_ == 0))
        fns = np.sum(np.logical_and(preds_ == 0, labels_ == 1))
        eps = 1e-8
        precision = tps / np.maximum(tps + fps, eps)
        recall = tps / np.maximum(tps + fns, eps)
        f1 = 2 * precision * recall / np.maximum(precision + recall, eps)
        return [f1, precision, recall]


class MaxF1(BaseMetric):
    """Maximum F1 score accross multiple thresholds."""

    name = "max_f1"
    score_names = ["max_f1", "max_f1_threshold", "max_f1_precision", "max_f1_recall"]
    is_higher_better = True
    need_pred_proba = True

    def _compute_scores(self, labels: Sequence, preds: Sequence) -> List[float]:
        return compute_maxf1_stats(labels, preds)


def compute_maxf1_stats(labels: Sequence, preds: Sequence) -> List[float]:
    """Computes max f1 and the related stats for binary classification.

    Args:
        labels: An iterable variable of binary labels
        preds: An iterable variable of probability predictions. preds are clipped into
            range [0, 1] before computing f1 score

    Returns:
        A tuple of (max f1, threshold at max f1, precision at max f1, recall at max f1).

    Raises:
        ValueError if labels contains non-binary values.
    """
    num_thresholds = 1000
    labels_ = np.array(labels).reshape((-1, 1))  # shape (N, 1)
    preds_ = np.clip(np.array(preds), 0, 1).reshape((-1, 1))  # shape (N, 1)
    _check_binary_list(labels)
    thresholds = np.arange(start=0, stop=1, step=1.0 / num_thresholds)  # shape (T,)
    binary_preds = preds_ >= thresholds  # shape (N, T) with T = num_thresholds
    tps = np.sum(np.logical_and(binary_preds == 1, labels_ == 1), axis=0)
    fps = np.sum(np.logical_and(binary_preds == 1, labels_ == 0), axis=0)
    fns = np.sum(np.logical_and(binary_preds == 0, labels_ == 1), axis=0)
    eps = 1e-8
    precisions = tps / np.maximum(tps + fps, eps)
    recalls = tps / np.maximum(tps + fns, eps)
    f1s = 2 * precisions * recalls / np.maximum(precisions + recalls, eps)
    max_f1 = np.max(f1s)
    max_f1_index = np.argmax(f1s)
    max_f1_threshold = thresholds[max_f1_index]
    max_f1_precision = precisions[max_f1_index]
    max_f1_recall = recalls[max_f1_index]
    return [max_f1, max_f1_threshold, max_f1_precision, max_f1_recall]


class SMAPE(BaseMetric):
    name = "smape"  # symmetric-mean-percentage-error
    score_names = ["smape"]
    is_higher_better = False

    def _compute_scores(self, labels: Sequence, preds: Sequence) -> List[float]:
        nominator = 2 * np.abs(np.array(preds) - np.array(labels))
        denominator = np.abs(labels) + np.abs(preds)
        return [100 * np.mean(np.divide(nominator, denominator))]  # type: ignore


class R2(BaseMetric):
    name = "r2"
    score_names = ["r2"]
    is_higher_better = True

    def _compute_scores(self, labels: Sequence, preds: Sequence) -> List[float]:
        return [sk_metrics.r2_score(labels, preds)]  # type: ignore


def get_instantiated_metric_dict() -> Dict[str, BaseMetric]:
    res = {}
    for sub_class in BaseMetric.__subclasses__():
        metric = sub_class()
        res[metric.name] = metric
    return res


def _check_binary_list(nums: Sequence) -> None:
    """Checks if a list only contains binary values.

    Raises an error if not.
    """
    unique_vals = np.unique(nums)
    for label in unique_vals:
        if label not in [0, 1]:
            raise ValueError(
                f"Input must contain only binary values, got {unique_vals}"
            )
