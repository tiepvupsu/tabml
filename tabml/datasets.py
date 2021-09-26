import tempfile
import zipfile
from pathlib import Path
from typing import Dict

import pandas as pd
from six.moves.urllib.request import urlretrieve

SUPPORTED_DATASETS = ("titanic", "california_housing", "movielen-1m")
DATA_SOURCE = "https://media.githubusercontent.com/media/tiepvupsu/tabml_data/master/"


def download_titanic() -> Dict[str, pd.DataFrame]:
    return {
        "train": pd.read_csv(DATA_SOURCE + "titanic/train.csv"),
        "test": pd.read_csv(DATA_SOURCE + "titanic/test.csv"),
        "gender_submission": pd.read_csv(DATA_SOURCE + "titanic/gender_submission.csv"),
    }


def download_california_housing() -> Dict[str, pd.DataFrame]:
    return {"housing": pd.read_csv(DATA_SOURCE + "california_housing/housing.csv")}


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
