from typing import Any, Dict, List, Union

import pydantic


class DataLoader(pydantic.BaseModel):
    cls_name: str = "tabml.data_loaders.BaseDataLoader"  # name of DataLoader class

    # path to feature_config
    feature_config_path: str = ""

    # list of features going in to the model
    features_to_model: List[str] = []

    # filters to determine training, validation, and submission dataset. For each
    # dataset, rows that are met all conditions will be selected.
    train_filters: List[str] = []
    validation_filters: List[str] = []
    submission_filters: List[str] = []

    # name of label column
    label_col: str = ""
    # name of feature to create sample weight.
    feature_to_create_weights: Union[str, None] = None


class ModelWrapper(pydantic.BaseModel):
    cls_name: str = "tabml.data_loaders.BaseDataLoader"  # name of model_wrapper class
    sklearn_cls: str = ""  # a sklearn-like model class
    model_params: Dict = {}
    fit_params: Dict = {}
    weight_params: Dict = {}


class ModelAnalysis(pydantic.BaseModel):
    # list of metrics to compute.
    # Each element must be the name of a subclass of tabml.metrics.BaseMetric
    metrics: List[str] = []

    # run shap explainer or not. This is expensive, use with care. The default is True
    # for backward compatibility.
    show_feature_importance: bool = True

    # How many datapoints used for running explainer.
    # Only used when show_feature_importance = True
    # int: number of datapoints
    # float: must be a number in (0, 1]
    # None: all datapoints
    explainer_train_size: Union[int, float, None] = None

    # list of features for analysis
    by_features: List[str] = []

    # This field specifies the label for analysis. In some cases, we use a
    # transformation of the true label for training but need to evaluate on the
    # original label.
    by_label: Union[str, None] = None

    # To avoid expensive computations, a subset of training data could be used instead.
    # This field specifies the subset size. Usage is similar to explainer_train_size.
    metric_train_size: Union[int, float, None] = None

    @pydantic.validator("explainer_train_size")
    def validate_float_explainer_train_size(cls, v):
        if isinstance(v, float) and (v <= 0 or v >= 1):
            raise ValueError(
                "When explainer_train_size is a float, it must be in (0, 1)."
            )
        return v.title()

    @pydantic.validator("metric_train_size")
    def validate_float_metric_train_size(cls, v):
        if isinstance(v, float) and (v <= 0 or v >= 1):
            raise ValueError("When metric_train_size is a float, it must be in (0, 1).")
        return v.title()


class PipelineConfig(pydantic.BaseModel):
    config_name: str
    data_loader: DataLoader
    model_wrapper: ModelWrapper
    model_analysis: ModelAnalysis


class ModelBundle(pydantic.BaseModel):
    pipeline_config: PipelineConfig
    model: Any
