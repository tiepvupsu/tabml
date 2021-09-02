from tabml.model_wrappers import LgbmRegressorModelWrapper
from tabml.pipelines import BasePipeline


class CustomLgbmRegressorModelWrapperLog10(LgbmRegressorModelWrapper):
    """Custom Model Wrapper for prediction log10 of median house value."""

    def predict(self, data):
        # In prediction, the model should return the outputs that are in the same
        # space with the true label.
        return 10 ** self.model.predict(data)


def train_lgbm():
    path_to_config = "configs/lgbm_config.pb"
    pipeline = BasePipeline(
        path_to_config, custom_model_wrapper=CustomLgbmRegressorModelWrapperLog10
    )
    pipeline.run()


if __name__ == "__main__":
    train_lgbm()
