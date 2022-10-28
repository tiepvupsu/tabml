import numpy as np

from tabml.utils.embedding import NearestNeighbor


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

        assert NearestNeighbor(items, "l2").find_nearest_neighbors(query, 1)[0] == 0
        assert NearestNeighbor(items, "dot").find_nearest_neighbors(query, 1)[0] == 1
        assert NearestNeighbor(items, "cosine").find_nearest_neighbors(query, 1)[0] == 2
