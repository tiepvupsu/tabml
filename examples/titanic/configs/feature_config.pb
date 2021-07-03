# proto-file: tabml/protos/feature_manager.proto
# proto-message: FeatureConfig

raw_data_dir {
  path: "examples/titanic/data"
  is_absolute_path: false
}

dataset_name: "sample"
base_features {
  name: "passenger_id"
  dtype: INT32
}
base_features {
  name: "sibsp"
  dtype: INT32
}
base_features {
  name: "parch"
  dtype: INT32
}
base_features {
  name: "fare"
  dtype: FLOAT
}

base_features {
  name: "age"
  dtype: FLOAT
}

transforming_features {
  index: 1
  name: "is_train"
  dependencies: "passenger_id"
  dtype: BOOL
}

transforming_features {
  index: 2
  name: "imputed_age"
  dtype: FLOAT
  dependencies: "age"
  dependencies: "is_train"
}

transforming_features {
  index: 3
  name: "bucketized_age"
  dependencies: "imputed_age"
  dtype: FLOAT
}

transforming_features {
  index: 5
  name: "survived"
  dtype: FLOAT
}

transforming_features {
  index: 6
  name: "sex"
  dtype: STRING
}

transforming_features {
  index: 7
  name: "coded_sex"
  dependencies: "sex"
  dtype: INT32
}

transforming_features {
  index: 8
  name: "pclass"
  dtype: FLOAT
}

transforming_features {
  index: 9
  name: "coded_pclass"
  dependencies: "pclass"
  dtype: INT32
}

transforming_features {
  index: 10
  name: "embarked"
  dtype: STRING
}

transforming_features {
  index: 11
  name: "coded_embarked"
  dependencies: "embarked"
  dtype: INT32
}

transforming_features {
  index: 12
  name: "title"
  dtype: STRING
}

transforming_features {
  index: 13
  name: "coded_title"
  dependencies: "title"
  dtype: INT32
}

transforming_features {
  index: 14
  name: "min_max_scaled_age"
  dependencies: "imputed_age"
  dependencies: "is_train"
  dtype: FLOAT
}
