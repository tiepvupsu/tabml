import random
from pathlib import Path

from qcore.asserts import AssertRaises, assert_eq

from tabml.utils import utils


class TestRandomString:
    def test_length(self):
        length = random.randint(1, 100)
        str1 = utils.random_string(length)
        # sanity check
        assert_eq(length, len(str1))

    def test_different_each_time(self):
        str1 = utils.random_string(50)
        str2 = utils.random_string(50)
        # check two calls return differnt strings.
        # There is an extremely low chance that these two random strings are equal
        assert_eq(2, len(set([str1, str2])))


class TestShowFeatureImportance:
    def test_1(self):
        data = {"a": 1, "b": 10, "c": 5}
        utils.show_feature_importance(data)

    def test_negative_importance(self):
        data = {"a": -1, "b": 0, "c": 1}
        with AssertRaises(ValueError) as assert_raises:
            _ = utils.show_feature_importance(data)

        assertion_error = assert_raises.expected_exception_found
        assert_eq(
            True,
            assertion_error.args[0].startswith(
                "All feature importances need to be non-negative, got data = "
            ),
        )


class TestCheckUniqness:
    def test_pass_if_unique(self):
        items = [1, 2, 3, 4]
        utils.check_uniqueness(items)

    def test_assert_if_not_unique(self):
        items = [1, 2, 3, 1]
        with AssertRaises(AssertionError) as assert_raises:
            utils.check_uniqueness(items)

        assertion_error = assert_raises.expected_exception_found
        assert_eq(
            "There are duplicate objects in the list: {1: 2}.", assertion_error.args[0]
        )


def test_save_and_load_pickle(tmp_path):
    an_object = random.randint(1, 100)
    filename = utils.random_string(4)
    utils.save_as_pickle(an_object, tmp_path, filename)
    another_object = utils.load_pickle(Path(tmp_path) / filename)
    assert_eq(an_object, another_object)
