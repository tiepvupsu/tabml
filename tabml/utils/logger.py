import sys

from loguru import logger

logger.add(
    sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO"
)


def lgbm_format_eval_result(value, show_stdv=True):
    if len(value) == 4:
        return "{0}'s {1}:{2:.5f}".format(value[0], value[1], value[2])
    elif len(value) == 5:
        if show_stdv:
            return "{0}'s {1}:{2:.5f}+{3:.5f}".format(
                value[0], value[1], value[2], value[4]
            )
        return "{0}'s {1}:{2:.5f}".format(value[0], value[1], value[2])
    raise ValueError("Wrong metric value")


def xgboost_format_eval_result(value, show_stdv=True):
    if len(value) == 2:
        return "{0}:{1:.5f}".format(value[0], value[1])
    if len(value) == 3:
        if show_stdv:
            return "{0}:{1:.5f}+{2:.5f}".format(value[0], value[1], value[2])
        return "{0}:{1:.5f}".format(value[0], value[1])
    raise ValueError("wrong metric value")


def boosting_logger_eval(model: str, period: int = 1, show_stdv: bool = True):
    """Creates a callback that prints the evaluation results for lgbm.

    Args:
        model (str): model name, lgbm or xgb
        period (int, optional):
            The period to print the evaluation results.
            Defaults to 1
        show_stdv (bool, optional):
            Whether to show stdv (if provided).
            Default to True

    Returns:
        function: The callback that prints the evaluation results
            every ``period`` iteration(s).
    """
    if model == "lgbm":
        format_result_func = lgbm_format_eval_result
    elif model == "xgboost":
        format_result_func = xgboost_format_eval_result
    else:
        raise TypeError(f"Unexpected model {model}. Only accept lgbm or xgb model.")

    def _callback(env):
        if (
            period > 0
            and env.evaluation_result_list
            and (env.iteration + 1) % period == 0
        ):
            result = "\t".join(
                [format_result_func(x, show_stdv) for x in env.evaluation_result_list]
            )
            logger.info(f"[{env.iteration}] {result}")

    return _callback
