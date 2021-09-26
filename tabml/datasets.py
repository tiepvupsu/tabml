import tempfile
import zipfile
from pathlib import Path
from typing import Dict

import pandas as pd
from six.moves.urllib.request import urlretrieve

from tabml.utils.logger import logger

SUPPORTED_DATASETS = ("titanic", "california_housing", "movielen-1m")
DATA_SOURCE = "https://media.githubusercontent.com/media/tiepvupsu/tabml_data/master/"


def _get_full_data_path(local_dir, tabml_data_dir, filename):
    """Returns the full path to a data file.

    Returns local path if exists, otherwise the tabml_data path.
    """
    if Path(local_dir).joinpath(filename).exists():
        return Path(local_dir).joinpath(filename)

    return f"{DATA_SOURCE}/{tabml_data_dir}/{filename}"


def load_titanic(data_dir: str) -> Dict[str, pd.DataFrame]:
    """Loads data from data_dir folder if they exist, downloads otherwise."""

    train_path = _get_full_data_path(data_dir, "titanic", "train.csv")
    test_path = _get_full_data_path(data_dir, "titanic", "test.csv")
    gender_submission_path = _get_full_data_path(
        data_dir, "titanic", "gender_submission.csv"
    )
    logger.info(
        f"Loading dataframes from {train_path}, {test_path}, and "
        f"{gender_submission_path}"
    )

    return {
        "train": pd.read_csv(train_path),
        "test": pd.read_csv(test_path),
        "gender_submission": pd.read_csv(gender_submission_path),
    }


def load_california_housing(data_dir: str) -> Dict[str, pd.DataFrame]:
    data_path = _get_full_data_path(data_dir, "california_housing", "housing.csv")
    logger.info(f"Loading dataframe from {data_path}")
    return {"housing": pd.read_csv(DATA_SOURCE + "california_housing/housing.csv")}


# TODO: manually download the dataset and load from disk
def download_movielen_1m() -> Dict[str, pd.DataFrame]:
    url = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
    # Download and extract ml-1m dataset to a temporary folder.
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_file_path = tmp_dir.joinpath("tmp.zip")
    urlretrieve(url, tmp_file_path)
    with zipfile.ZipFile(tmp_file_path, "r") as tmp_zip:
        tmp_zip.extractall(tmp_dir)

    users = pd.read_csv(
        tmp_dir.joinpath("ml-1m", "users.dat"),
        delimiter="::",
        engine="python",
        names=["UserID", "Gender", "Age", "Occupation", "Zip-code"],
    )
    movies = pd.read_csv(
        tmp_dir.joinpath("ml-1m", "movies.dat"),
        delimiter="::",
        encoding="ISO-8859-1",
        engine="python",
        names=["MovieID", "Title", "Genres"],
    )

    ratings = pd.read_csv(
        tmp_dir.joinpath("ml-1m", "ratings.dat"),
        delimiter="::",
        engine="python",
        names=["UserID", "MovieID", "Rating", "Timestamp"],
        usecols=["UserID", "MovieID", "Rating", "Timestamp"],
    )

    return {"users": users, "movies": movies, "ratings": ratings}
