import random

from qcore.asserts import assert_eq

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
