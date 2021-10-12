import numpy as np
from qcore.asserts import assert_eq

from tabml.utils import embedding


class TestNearestNeighbor:
    def test_1(self):
        query = np.array([1, 2])
        items = np.array(
            [
                [1, 2.2],  # neareast in l2
                [10, 21],  # neareast in dot product similarity
                [2, 4],  # nearest in cosine similarity
            ]
        )

        assert_eq(
            embedding.NearestNeighbor(items, "l2").find_nearest_neighbors(query, 1)[0],
            0,
        )
        assert_eq(
            embedding.NearestNeighbor(items, "dot").find_nearest_neighbors(query, 1)[0],
            1,
        )
        assert_eq(
            embedding.NearestNeighbor(items, "cosine").find_nearest_neighbors(query, 1)[
                0
            ],
            2,
        )
