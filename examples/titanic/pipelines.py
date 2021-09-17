from tabml.pipelines import BasePipeline


def run(path_to_config: str):
    pipeline = BasePipeline(path_to_config)
    pipeline.run()


def train_lgbm():
    path_to_config = "configs/lgbm_config.yaml"
    run(path_to_config)


def train_xgboost():
    path_to_config = "configs/xgboost_config.yaml"
    run(path_to_config)


def train_catboost():
    path_to_config = "configs/catboost_config.yaml"
    run(path_to_config)


if __name__ == "__main__":
    train_lgbm()
