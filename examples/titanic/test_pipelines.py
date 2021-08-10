import os
import sys

from . import pipelines

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, os.pardir))
sys.path.insert(0, PROJECT_DIR)


def test_full_pipeline():
    pipelines.train_lgbm()
