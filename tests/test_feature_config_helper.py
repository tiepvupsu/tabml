import pytest
from qcore.asserts import AssertRaises, assert_eq

from tabml import feature_config_helper
from tabml.utils.pb_helpers import parse_feature_config_pb
from tabml.utils.utils import write_str_to_file


class TestFeatureConfigHelper:
    @pytest.fixture(autouse=True)
    def setup_class(cls, tmp_path):
        feature_config_str = """
            raw_data_dir: "dummy"
            dataset_name: "dummy"
            base_features {
              name: "a"
              dtype: STRING
            }
            transforming_features {
              index: 1
              name: "b"
              dependencies: "a"
            }
            transforming_features {
              index: 2
              name: "c"
              dependencies: "a"
              dependencies: "b"
            }
            transforming_features {
              index: 3
              name: "d"
              dependencies: "a"
            }
            transforming_features {
              index: 4
              name: "e"
              dependencies: "c"
            }
        """
        pb_config_path = tmp_path / "feature_config_str.pbtxt"
        write_str_to_file(feature_config_str, pb_config_path)
        cls.fm_helper = feature_config_helper.FeatureConfigHelper(pb_config_path)

    def test_raise_value_error_with_invalid_indexes(self, tmp_path):
        invalid_index_pb_str = """
            # invalid config with indexes are not continuous
            raw_data_dir: "dummy"
            dataset_name: "dummy"
            base_features {
              name: "TIME"
              dtype: DATETIME
            }
            transforming_features {
              index: 1
              name: "weekday"
              dependencies: "TIME"
            }
            transforming_features {
              index: 1
              name: "hour"
              dependencies: "TIME"
            }
        """
        pb_config_path = tmp_path / "tmp.pb"
        write_str_to_file(invalid_index_pb_str, pb_config_path)
        with AssertRaises(ValueError) as assert_raises:
            feature_config_helper.FeatureConfigHelper(pb_config_path)

        error_message = assert_raises.expected_exception_found
        assert_eq(
            True,
            error_message.args[0].startswith(
                "Feature indexes must be a list of increasing positive integers. "
                "Got indexes = [1, 1]"
            ),
        )

    def test_raise_assertion_error_with_duplicate_features(self, tmp_path):
        pb_str = """
            raw_data_dir: "dummy"
            dataset_name: "dummy"
            base_features {
              name: "TIME"
              dtype: DATETIME
            }
            transforming_features {
              index: 1
              name: "weekday"
              dependencies: "TIME"
            }
            transforming_features {
              index: 2
              name: "weekday"
            }
        """
        pb_config_path = tmp_path / "tmp.pb"
        write_str_to_file(pb_str, pb_config_path)
        with AssertRaises(AssertionError) as assert_raises:
            feature_config_helper.FeatureConfigHelper(pb_config_path)

        error_message = assert_raises.expected_exception_found
        assert_eq(
            True,
            error_message.args[0].startswith(
                "There are duplicate objects in the list: "
            ),
        )

    def test_raise_value_error_with_invalid_dependencies(self, tmp_path):
        invalid_dependency_pb_str = """
            raw_data_dir: "dummy"
            dataset_name: "dummy"
            base_features {
              name: "TIME"
              dtype: DATETIME
            }
            transforming_features {
              index: 1
              name: "weekday"
              dependencies: "date"
            }
        """
        pb_config_path = tmp_path / "tmp.pb"
        write_str_to_file(invalid_dependency_pb_str, pb_config_path)
        with AssertRaises(AssertionError) as assert_raises:
            feature_config_helper.FeatureConfigHelper(pb_config_path)

        error_message = assert_raises.expected_exception_found
        assert_eq(
            True,
            error_message.args[0].startswith(
                "Feature weekday depends on feature date that is undefined."
            ),
        )

    def test_find_dependents(self, tmp_path):
        got_1 = self.fm_helper.find_dependents("a")
        expected_1 = ["b", "c", "d", "e"]
        assert_eq(expected_1, got_1)

        got_2 = self.fm_helper.find_dependents("b")
        expected_2 = ["c", "e"]
        assert_eq(expected_2, got_2)

        got_3 = self.fm_helper.find_dependents("d")
        expected_3 = []
        assert_eq(expected_3, got_3)

    def test_append_dependents(self, tmp_path):
        got = self.fm_helper.append_dependents(["d", "b"])
        expected = ["b", "c", "d", "e"]
        assert_eq(expected, got)

    def test_extract_config_1(self, tmp_path):
        subset_features = ["e"]
        expected_pb_str = """
            raw_data_dir: "dummy"
            dataset_name: "dummy"
            base_features {
              name: "a"
              dtype: STRING
            }
            transforming_features {
              index: 1
              name: "b"
              dependencies: "a"
            }
            transforming_features {
              index: 2
              name: "c"
              dependencies: "a"
              dependencies: "b"
            }
            transforming_features {
              index: 4
              name: "e"
              dependencies: "c"
            }
        """
        new_pb_config_path = tmp_path / "new_tmp.pb"
        write_str_to_file(expected_pb_str, new_pb_config_path)
        new_config = self.fm_helper.extract_config(selected_features=subset_features)
        assert_eq(parse_feature_config_pb(new_pb_config_path), new_config)

    def test_extract_config_2(self, tmp_path):
        subset_features = ["d"]
        expected_pb_str = """
            raw_data_dir: "dummy"
            dataset_name: "dummy"
            base_features {
              name: "a"
              dtype: STRING
            }
            transforming_features {
              index: 3
              name: "d"
              dependencies: "a"
            }
        """
        new_pb_config_path = tmp_path / "new_tmp.pb"
        write_str_to_file(expected_pb_str, new_pb_config_path)
        new_config = self.fm_helper.extract_config(selected_features=subset_features)
        assert_eq(parse_feature_config_pb(new_pb_config_path), new_config)

    def test_raise_value_error_with_invalid_feature_to_extract(self, tmp_path):
        subset_features = ["a", "y", "z"]
        with AssertRaises(ValueError) as assert_raises:
            self.fm_helper.extract_config(selected_features=subset_features)

        error_message = assert_raises.expected_exception_found
        assert_eq(
            error_message.args[0], "Features ['y', 'z'] are not in the original config."
        )
