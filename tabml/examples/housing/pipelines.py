from tabml.pipelines import BasePipeline


def train_lgbm():
    path_to_config = "configs/lgbm_config.yaml"
    pipeline = BasePipeline.from_config_path(path_to_config)
    pipeline.run()


def train_xgboost():
    path_to_config = "configs/xgboost_config.yaml"
    pipeline = BasePipeline.from_config_path(path_to_config)
    pipeline.run()


def train_catboost():
    path_to_config = "configs/catboost_config.yaml"
    pipeline = BasePipeline.from_config_path(path_to_config)
    pipeline.run()


if __name__ == "__main__":
    train_lgbm()
