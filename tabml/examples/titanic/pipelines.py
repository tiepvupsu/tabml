from tabml.pipelines import BasePipeline


def run(pipeline_config_path: str):
    pipeline = BasePipeline(config=pipeline_config_path)
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
