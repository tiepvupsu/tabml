from tabml.pipelines import BasePipeline


def train_lgbm():
    path_to_config = "configs/lgbm_config.yaml"
    pipeline = BasePipeline(path_to_config)
    pipeline.run()
    return pipeline


def train_xgboost():
    path_to_config = "configs/xgboost_config.yaml"
    pipeline = BasePipeline(path_to_config)
    pipeline.run()
    return pipeline


def train_catboost():
    path_to_config = "configs/catboost_config.yaml"
    pipeline = BasePipeline(path_to_config)
    pipeline.run()
    return pipeline


if __name__ == "__main__":
    train_lgbm()
