import os
import pickle
import random
import string
import subprocess
import tempfile
import typing
from collections import Counter
from pathlib import Path
from typing import Any, Collection

import git
import GPUtil

from tabml.utils.logger import logger


def random_string(length: int = 10) -> str:
    """Returns a random string of lowercase letters and digits with a given length.

    Args:
        length: output length

    Returns:
        a random string with length being `length`
    """
    return "".join(
        [random.choice(string.ascii_letters + string.digits) for n in range(length)]
    )


def write_str_to_file(a_str: str, filename: str) -> None:
    """Writes a string to a file."""
    with open(filename, "w") as str_file:
        str_file.write(a_str)


def check_uniqueness(items: Collection) -> None:
    """Checks if an array containing unique elements.

    Args:
        items: A list of objects.

    Returns:
        Does not return anything. If this function passes, it means that all objects
        are unique.

    Raises:
        Assertion error with list of duplicate objects.
    """
    counter = Counter(items)
    duplicates = {item: count for item, count in counter.items() if count > 1}
    assert not duplicates, f"There are duplicate objects in the list: {duplicates}."


def get_git_root():
    """Gets the root path of the current git repo."""
    git_repo = git.Repo(Path.cwd(), search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root


def show_feature_importance(data: typing.Dict[str, float]) -> None:
    """
    Shows feature importance in terminal.

    Importances are shown in desecending order.
    Note that termgraph (a tool for visualize graph in terminal) only shows meaningful
    graphs with positive values. Fortunately, XGB model only outputs positive feature
    importances. If LGBM and Keras, if any, have negative feature importance, then a
    quick modification is to visualize absolute values. In fact, the magnitude of
    feature importances are more _important_ than their actual values.

    Args:
        data: a dictionary of {feature: imporatnce}

    Raises:
        ValueError if any importance is negative.
    """
    assert data is not None, "input dictionary can not be empty"
    tmp_file = tempfile.NamedTemporaryFile()
    # sort feature by desecending importance
    feature_importance_tuples = sorted(data.items(), key=lambda x: -x[1])

    if feature_importance_tuples[-1][-1] < 0:
        raise ValueError(
            f"All feature importances need to be non-negative, got data = {data}"
        )

    # write to file
    # TODO: find a way to visualize data directly rather than saving it in an
    # intermediate file.
    with open(tmp_file.name, "w") as fout:
        for feature, importance in feature_importance_tuples:
            fout.write(f"{feature}, {importance}\n")

    logger.info("Feature importance:")
    logger.info(subprocess.getoutput(f"termgraph {tmp_file.name}"))


def save_as_pickle(an_object: Any, path: str, filename: str) -> None:
    """Saves an object as a pickle file.

    Args:
        an_object: A python object. Can be list, dict etc.
        path: The path where to save.
        filename: The filename
    """
    if not Path(path).exists():
        Path(path).mkdir(parents=True, exist_ok=True)

    file_path = Path(path) / filename
    with open(file_path, "wb") as handle:
        pickle.dump(an_object, handle)
    logger.info(f"File is saved successfully to {file_path}.")


def load_pickle(path_to_obj):
    # open a file, where you stored the pickled data
    file = open(path_to_obj, "rb")
    # dump information to that file
    obj = pickle.load(file)
    # close the file
    file.close()

    return obj


def is_gpu_available():
    return (
        len(
            GPUtil.getAvailable(
                order="first",
                limit=1,
                maxLoad=0.5,
                maxMemory=0.5,
                includeNan=False,
                excludeID=[],
                excludeUUID=[],
            )
        )
        > 0
    )


def change_working_dir_pytest(func):
    # https://stackoverflow.com/a/62055409/11871829
    def apply(request):
        os.chdir(request.fspath.dirname)
        func()
        os.chdir(request.config.invocation_dir)

    return apply
