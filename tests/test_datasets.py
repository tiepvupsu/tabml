from tabml import datasets


def test_download_movielen_1m():
    datasets.download_as_dataframes("movielen-1m")
