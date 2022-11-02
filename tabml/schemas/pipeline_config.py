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


class ModelWrapper(pydantic.BaseModel):
    cls_name: str = "tabml.data_loaders.BaseDataLoader"  # name of model_wrapper class
    sklearn_cls: str = ""  # a sklearn-like model class
    model_params: Dict = {}
    fit_params: Dict = {}


class ModelAnalysis(pydantic.BaseModel):
    # list of metrics to compute.
    # Each element must be the name of a subclass of tabml.metrics.BaseMetric
    metrics: List[str] = []

    # run shap explainer or not. This is expensive, use with care. The default is True
    # for backward compatibility.
    show_feature_importance: bool = True

    # list of features for analysis
    by_features: List[str] = []

    # This field specifies the label for analysis. In some cases, we use a
    # transformation of the true label for training but need to evaluate on the
    # original label.
    by_label: Union[str, None] = None

    # To avoid expensive computations, e.g. computing SHAP values or inference,
    # a subset of training data could be used instead. This field specifies size
    # of that subset.
    training_size: Union[int, None] = None


class PipelineConfig(pydantic.BaseModel):
    config_name: str
    data_loader: DataLoader
    model_wrapper: ModelWrapper
    model_analysis: ModelAnalysis


class ModelBundle(pydantic.BaseModel):
    pipeline_config: PipelineConfig
    model: Any
