from tabml.pipelines import BasePipeline


def train_lgbm():
    path_to_config = "configs/lgbm_config.pb"
    pipeline = BasePipeline(path_to_config)
    pipeline.run()


if __name__ == "__main__":
    train_lgbm()
