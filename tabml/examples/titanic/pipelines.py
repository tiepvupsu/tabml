from pathlib import Path
from typing import Union

from tabml.pipelines import BasePipeline
from tabml.schemas.pipeline_config import PipelineConfig


def run(config: Union[str, Path, PipelineConfig]):
    pipeline = BasePipeline(config=config)
    pipeline.run()


def train_lgbm():
    pipeline_config_path = "configs/lgbm_config.yaml"
    run(pipeline_config_path)


def train_xgboost():
    pipeline_config_path = "configs/xgboost_config.yaml"
    run(pipeline_config_path)


def train_catboost():
    pipeline_config_path = "configs/catboost_config.yaml"
    run(pipeline_config_path)


def train_randomforest():
    pipeline_config_path = "./configs/rf_config.yaml"
    run(pipeline_config_path)


if __name__ == "__main__":
    train_lgbm()
