# proto-file: tabml/protos/feature_manager.proto
# proto-message: FeatureConfig

raw_data_dir: "tabml/titanic/data"
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

transforming_features {
  index: 1
  name: "age"
  dtype: FLOAT
}

transforming_features {
  index: 2
  name: "bucketized_age"
  dependencies: "age"
  dtype: FLOAT
}

transforming_features {
  index: 4
  name: "survived"
  dtype: FLOAT
}

transforming_features {
  index: 5
  name: "sex"
  dtype: STRING
}

transforming_features {
  index: 6
  name: "coded_sex"
  dependencies: "sex"
  dtype: INT32
}

transforming_features {
  index: 7
  name: "pclass"
  dtype: FLOAT
}

transforming_features {
  index: 8
  name: "coded_pclass"
  dependencies: "pclass"
  dtype: INT32
}

transforming_features {
  index: 9
  name: "embarked"
  dtype: STRING
}

transforming_features {
  index: 10
  name: "coded_embarked"
  dependencies: "embarked"
  dtype: INT32
}

transforming_features {
  index: 11
  name: "title"
  dtype: STRING
}

transforming_features {
  index: 12
  name: "coded_title"
  dependencies: "title"
  dtype: INT32
}
