from tabml.housing import pipelines


def test_full_pipeline():
    pipelines.train_lgbm()
