import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch.multiprocessing
import torch.nn.functional as F
import tqdm

from tabml.utils.logger import logger

MIN_PRODUCT_PER_ORDER = 5
MIN_PRODUCT_FREQUENCY = 10


def get_product_name_by_id() -> Dict[int, str]:
    logger.info("Load product dataframe...")
    df = pd.read_csv("data/products.csv", usecols=["product_id", "product_name"])
    return df.set_index("product_id").to_dict()["product_name"]


def get_list_orders(testing: bool = True) -> List[List[int]]:
    logger.info("Load order dataframe...")

    if testing:
        order_df = pd.read_csv("data/order_products__train.csv")
    else:
        order_prior_df = pd.read_csv("data/order_products__prior.csv")
        order_train_df = pd.read_csv("data/order_products__train.csv")
        order_df = pd.concat([order_prior_df, order_train_df], axis=0)
    order_df = order_df.sort_values(by=["order_id", "add_to_cart_order"])
    return order_df.groupby("order_id")["product_id"].apply(list).tolist()


def filter_small_orders(all_orders: List[List[int]], min_product_per_order: int = 10):
    orders = [order for order in all_orders if len(order) >= min_product_per_order]
    logger.info(
        f"Number of orders with at least {min_product_per_order}: {len(orders)}"
    )
    return orders


def run(testing=True):
    product_name_by_id = get_product_name_by_id()
    all_orders = get_list_orders(testing)
    orders = filter_small_orders(
        all_orders, min_product_per_order=MIN_PRODUCT_PER_ORDER
    )


if __name__ == "__main__":
    run()
