# TabML notes

# Feature config

- [ ] Find the independet trees in feature graph for parallel processing. Two different subgraphs must not have any common nodes other than base features.

- [ ] Support group features
- [ ] Support multivalent features, e.g. onehot.
# Pipeline config

- [ ] Move model-dependent components in ModelAnalysis back to ModelWrapper.
  - [ ] Feature importance
    - TabNet is currently not supported by SHAP (but pytoched is). We can replace SHAP by other generic feature importance technique.
- [ ] Use unsupervised mode in TabNet
- [ ] Skip `cls_name` in `model_wrapper` part in yaml. One idea is to use a universal modelwrapper with different `model_cls`. **This might not be feasible if different models use different model_wrappers.

- [ ] Support Ranker problem

- [ ] Try different problems from Kaggle

- [ ] How to test problems with large datasets. One way is to only test with a fraction of data.

# Miscs

- [ ] Replace os path by pathlib.Path
