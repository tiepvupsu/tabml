import pandas as pd

SUPPORTED_DATASETS = ("titanic", "california_housing", "movielen-1m")
DATA_SOURCE = "https://media.githubusercontent.com/media/tiepvupsu/tabml_data/master/"


def download_as_dataframes(dataset_name: str):
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"{dataset_name} is not supported. Available datasets: {SUPPORTED_DATASETS}"
        )

    return {
        "titanic": download_titanic(),
        "california_housing": download_california_housing(),
        "movielen-1m": download_movielen_1m(),
    }[dataset_name]


def download_titanic():
    return (
        pd.read_csv(DATA_SOURCE + "titanic/train.csv"),
        pd.read_csv(DATA_SOURCE + "titanic/test.csv"),
        pd.read_csv(DATA_SOURCE + "titanic/gender_submission.csv"),
    )


def download_california_housing():
    return pd.read_csv(DATA_SOURCE + "california_housing/housing.csv")


def download_movielen_1m():
    users = pd.read_csv(
        DATA_SOURCE + "movielens/ml-1m/users.dat",
        delimiter="::",
        engine="python",
        names=["UserID", "Gender", "Age", "Occupation", "Zip-code"],
    )
    movies = pd.read_csv(
        DATA_SOURCE + "movielens/ml-1m/movies.dat",
        delimiter="::",
        encoding="ISO-8859-1",
        engine="python",
        names=["MovieID", "Title", "Genres"],
    )

    ratings = pd.read_csv(
        DATA_SOURCE + "movielens/ml-1m/ratings.dat",
        delimiter="::",
        engine="python",
        names=["UserID", "MovieID", "Rating", "Timestamp"],
        usecols=["UserID", "MovieID", "Rating", "Timestamp"],
    )

    return users, movies, ratings
