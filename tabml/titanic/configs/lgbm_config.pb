config_name: "lgbm"
data_loader {
  cls_name: "tabml.data_loaders.BaseDataLoader"
  feature_manager_config_path {path: "tabml/titanic/configs/feature_config.pb"}

  features_to_model: "coded_sex"
  features_to_model: "imputed_age"
  features_to_model: "bucketized_age"
  features_to_model: "min_max_scaled_age"
  features_to_model: "coded_pclass"
  features_to_model: "coded_title"

  label_col: "survived"

  train_filters: "is_train"
  validation_filters: "not is_train"
  validation_filters: "passenger_id <= 891"
  submission_filters: "passenger_id > 891"
}

model_wrapper {
  cls_name: "tabml.model_wrappers.LgbmClassifierModelWrapper"
  lgbm_params {learning_rate: 0.01}
}
trainer {
  cls_name: "tabml.trainers.LgbmTrainer"
}
model_analysis {
  metrics: "accuracy_score"
  metrics: "roc_auc"
  metrics: "max_f1"
  by_features: "sex"
  by_features: "pclass"
}
