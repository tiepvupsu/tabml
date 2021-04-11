# proto-file: tabml/protos/feature_manager.proto
# proto-message: FeatureConfig

raw_data_dir: "tabml/housing/data"
dataset_name: "processed"
base_features {
  name: "index"
  dtype: INT32
}

transforming_features {
  name: "housing_median_age"
  index: 1
  dtype: INT32
}

transforming_features {
  name: "total_rooms"
  index: 2
  dtype: INT32
}

transforming_features {
  name: "population"
  index: 3
  dtype: INT32
}
