import random
import string
from typing import Any, Collection, Set


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
    seen_items: Set[Any] = set()
    duplicates = set()
    for item in items:
        if item in seen_items:
            duplicates.add(item)
        seen_items.add(item)
    assert not duplicates, f"There are duplicate objects in the list: {duplicates}."
