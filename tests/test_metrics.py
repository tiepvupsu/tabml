import typing

import numpy as np
from qcore.asserts import AssertRaises, assert_eq

from tabml import metrics


class DummyMetric1(metrics.BaseMetric):
    name = "dummy1"
    score_names = ["a"]

    def _compute_scores(
        self, labels: typing.Collection, preds: typing.Collection
    ) -> typing.List[float]:
        return [0]


class DummyMetric2(metrics.BaseMetric):
    name = "dummy1"
    is_higher_better = True
    score_names = ["a"]

    def _compute_scores(
        self, labels: typing.Collection, preds: typing.Collection
    ) -> typing.List[float]:
        return [0]


class DummyMetric3(metrics.BaseMetric):
    name = "dummy1"
    is_higher_better = True
    score_names = ["a"]

    def _compute_scores(
        self, labels: typing.Collection, preds: typing.Collection
    ) -> typing.List[float]:
        return [0, 0]


class TestBaseMetric:
    def test_raise_ve_if_is_higher_better_not_defind(self):
        with AssertRaises(ValueError) as assert_raises:
            DummyMetric1().compute_scores([0], [0])

        error_message = assert_raises.expected_exception_found
        assert_eq(
            True,
            error_message.args[0].startswith(
                "Subclasses must define is_higher_better."
            ),
        )

    def test_raise_ve_if_labels_and_preds_have_diff_lens(self):
        with AssertRaises(ValueError) as assert_raises:
            DummyMetric2().compute_scores([0, 0], [0])

        error_message = assert_raises.expected_exception_found
        assert_eq(
            True,
            error_message.args[0].startswith("labels (len = 2) and preds (len = 1)"),
        )

    def test_score_names_and_compute_scores_have_diff_lens(self):
        with AssertRaises(ValueError) as assert_raises:
            DummyMetric3().compute_scores([0], [0])

        error_message = assert_raises.expected_exception_found
        assert_eq(
            True,
            error_message.args[0].startswith(
                "self.score_names (['a']) and scores ([0, 0]) must have the same length"
            ),
        )


class TestComputeMaxF1AndThreshold:
    def test_raise_if_labels_not_binary(self):
        labels = [0, 2]
        preds = [0, 1]
        with AssertRaises(ValueError) as assert_raises:
            metrics.compute_maxf1_stats(labels, preds)

        error_message = assert_raises.expected_exception_found
        assert_eq(
            True,
            error_message.args[0].startswith(
                "Input must contain only binary values, got "
            ),
        )

    def test_1(self):
        labels = [0, 0, 1, 1]
        preds = [0.1, 0.6991, 0.7009, 0.9]
        expected_max_f1 = 1
        expected_max_f1_threshold = 0.7
        expected_max_f1_precision = 1
        expected_max_f1_recall = 1
        max_f1, threshold, precision, recall = metrics.compute_maxf1_stats(
            labels, preds
        )
        assert_eq(expected_max_f1, max_f1)
        assert_eq(expected_max_f1_threshold, threshold, tolerance=1e-6)
        assert_eq(expected_max_f1_precision, precision)
        assert_eq(expected_max_f1_recall, recall)

    def test_2(self):
        labels = [0, 0, 1, 1]
        preds = [0.9, 0.7, 0.3, 0.1]
        expected_max_f1 = 2 / 3  # pr = 1/2, recall = 1
        expected_max_f1_precision = 1 / 2
        expected_max_f1_recall = 1
        max_f1, _, precision, recall = metrics.compute_maxf1_stats(labels, preds)
        assert_eq(expected_max_f1, max_f1)
        assert_eq(expected_max_f1_precision, precision)
        assert_eq(expected_max_f1_recall, recall)


class TestRocAuc:
    def test_return_nan_if_uniqe_label(self):
        got = metrics.RocAreaUnderTheCurve().compute_scores([0, 0], [0.5, 0.5])
        expected = {"roc_auc": np.nan}
        assert_eq(expected, got)


class TestF1Score:
    def test_computation(self):
        labels = [1, 0, 1, 1, 0]
        preds = [0, 0, 1, 1, 1]
        expected = {"f1": 2 / 3, "precision": 2 / 3, "recall": 2 / 3}
        got = metrics.F1Score().compute_scores(labels, preds)
        assert_eq(expected, got)


class TestSmape:
    def test_computation(self):
        labels = [1, 100, 1000]
        preds = [10, 10, 800]
        expected = 200 / 3 * (9 / 11 + 90 / 110 + 200 / 1800)
        got = metrics.SMAPE().compute_scores(labels, preds)["smape"]
        np.testing.assert_almost_equal(expected, got)


def test_metrics_have_unique_name():
    metric_by_name = metrics.get_instantiated_metric_dict()
    metric_names = metric_by_name.keys()
    assert_eq(len(metric_names), len(set(metric_names)))
