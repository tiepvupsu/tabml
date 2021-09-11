# proto-file: tabml/protos/pipeline.proto
# proto-message: Config
config_name: "lgbm"
data_loader {
    cls_name: "tabml.data_loaders.BaseDataLoader"
    feature_manager_config_path: "configs/feature_config.pb"

    features_to_model: "scaled_housing_median_age"
    features_to_model: "median_income"
    features_to_model: "scaled_total_rooms"
    features_to_model: "scaled_total_bedrooms"
    features_to_model: "scaled_population"
    features_to_model: "bucketized_latitude"
    features_to_model: "bucketized_longitude"
    features_to_model: "hashed_bucketized_latitude_X_bucketized_longitude"
    features_to_model: "encoded_ocean_proximity"

    label_col: "log10_median_house_value"

    train_filters: "is_train"

    validation_filters: "not is_train"

    submission_filters: "not is_train"
}
model_wrapper {
    # use a custom lgbm model_wrapper
    lgbm_params {
        learning_rate: 0.1
        n_estimators: 200
        objective: "rmse"
    }
}
trainer {
    cls_name: "tabml.trainers.LgbmTrainer"
    lgbm_params {
        categorical_feature: "encoded_ocean_proximity"
        categorical_feature: "hashed_bucketized_latitude_X_bucketized_longitude"
        categorical_feature: "bucketized_latitude"
        categorical_feature: "bucketized_longitude"
        eval_metric: "rmse"
    }
}
model_analysis {
    metrics: "smape"
    metrics: "rmse"
    by_features: "ocean_proximity"
    by_label: "median_house_value"
}
