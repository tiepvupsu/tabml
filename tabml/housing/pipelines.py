from tabml.pipelines import BasePipeline


def train_lgbm():
    path_to_config = "tabml/housing/configs/lgbm_config.pb"
    pipeline = BasePipeline(path_to_config)
    pipeline.run()
    pipeline.analyze_model()


def train_tabnet():
    path_to_config = "tabml/housing/configs/tabnet_config.pb"
    pipeline = BasePipeline(path_to_config)
    pipeline.run()
    pipeline.analyze_model()


if __name__ == "__main__":
    # train_lgbm()
    train_tabnet()
