# proto-file: tabml/protos/feature_manager.proto
# proto-message: FeatureConfig

raw_data {data_dir: "tabml/housing/data" is_absolute_path: false}
dataset_name: "processed"

base_features {
  name: "median_house_value"
  dtype: FLOAT
}

base_features {
  name: "housing_median_age"
  dtype: FLOAT
}

base_features {
  name: "total_rooms"
  dtype: FLOAT
}

transforming_features {
  name: "is_train"
  index: 1
  dtype: BOOL
}

transforming_features {
  name: "scaled_housing_median_age"
  index: 2
  dtype: FLOAT
  dependencies: "is_train"
  dependencies: "housing_median_age"
}

transforming_features {
  name: "scaled_clean_total_rooms"
  index: 3
  dtype: INT32
  dependencies: "is_train"
  dependencies: "total_rooms"
}

# transforming_features {
#   name: "population"
#   index: 4
#   dtype: INT32
# }
