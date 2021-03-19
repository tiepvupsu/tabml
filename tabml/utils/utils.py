import datetime
import random
import string


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


def current_time_str() -> str:
    now = datetime.now()
    return now.strftime("%y%m%d_%H%M%S")
