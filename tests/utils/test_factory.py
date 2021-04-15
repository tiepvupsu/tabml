from qcore.asserts import assert_eq

from tabml.feature_manager import BaseFeatureManager
from tabml.utils import factory


class TestCreate:
    def test_1(self):
        cls_name = "tabml.feature_manager.BaseFeatureManager"
        got = factory.create(cls_name)
        assert_eq(BaseFeatureManager, got)
