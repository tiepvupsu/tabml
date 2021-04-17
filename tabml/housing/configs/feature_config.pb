# proto-file: tabml/protos/feature_manager.proto
# proto-message: FeatureConfig

raw_data_dir {path: "tabml/housing/data" is_absolute_path: false}
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

base_features {
  name: "population"
  dtype: FLOAT
}

base_features {
  name: "total_bedrooms"
  dtype: FLOAT
}

base_features {
  name: "households"
  dtype: FLOAT
}

base_features {
  name: "median_income"
  dtype: FLOAT
}

base_features {
  name: "ocean_proximity"
  dtype: STRING
}
base_features {
  name: "longitude"
  dtype: FLOAT
}

base_features {
  name: "latitude"
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
  dtype: FLOAT
  dependencies: "is_train"
  dependencies: "total_rooms"
}

transforming_features {
  name: "scaled_clean_population"
  index: 4
  dtype: FLOAT
  dependencies: "is_train"
  dependencies: "population"
}

transforming_features {
  name: "scaled_clean_total_bedrooms"
  index: 5
  dtype: FLOAT
  dependencies: "is_train"
  dependencies: "total_bedrooms"
}

transforming_features {
  name: "scaled_clean_households"
  index: 6
  dtype: FLOAT
  dependencies: "is_train"
  dependencies: "households"
}

transforming_features {
  name: "scaled_clean_median_income"
  index: 7
  dtype: FLOAT
  dependencies: "is_train"
  dependencies: "median_income"
}

transforming_features {
  name: "log10_median_house_value"
  dtype: FLOAT
  index: 8
  dependencies: "median_house_value"
}

transforming_features {
  name: "bucketized_longitude"
  index: 9
  dtype: INT32
  dependencies: "longitude"
}

transforming_features {
  name: "bucketized_latitude"
  index: 10
  dtype: INT32
  dependencies: "latitude"
}

transforming_features {
  name: "bucketized_latitude_X_bucketized_longitude"
  index: 11
  dtype: STRING
  dependencies: "bucketized_latitude"
  dependencies: "bucketized_longitude"
}

transforming_features {
  name: "hashed_bucketized_latitude_X_bucketized_longitude"
  index: 12
  dtype: INT32
  dependencies: "bucketized_latitude_X_bucketized_longitude"
}
