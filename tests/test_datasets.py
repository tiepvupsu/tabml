from tabml import datasets


def test_download_titanic():
    datasets.download_as_dataframes("titanic")


def test_download_california_housing():
    datasets.download_as_dataframes("california_housing")


def test_download_movielen_1m():
    datasets.download_as_dataframes("movielen-1m")
