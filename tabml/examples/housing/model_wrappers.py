from tabml.model_wrappers import (
    CatBoostRegressorModelWrapper,
    LgbmRegressorModelWrapper,
    XGBoostRegressorModelWrapper,
)


class CustomModelWrapperLog10:
    def predict(self, data):
        # In prediction, the model should return the outputs that are in the same
        # space with the true label.
        return 10 ** self.model.predict(data)


class CustomLgbmRegressorModelWrapperLog10(
    LgbmRegressorModelWrapper, CustomModelWrapperLog10
):
    pass


class CustomXGBoostRegressorModelWrapperLog10(
    XGBoostRegressorModelWrapper, CustomModelWrapperLog10
):
    pass


class CustomCatBoostRegressorModelWrapperLog10(
    CatBoostRegressorModelWrapper, CustomModelWrapperLog10
):
    pass
