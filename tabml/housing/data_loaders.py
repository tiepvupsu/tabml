from tabml.data_loaders import BaseDataLoader


class HousingDataLoader(BaseDataLoader):
    def __init__(self, config):
        super().__init__(config)

    def get_val_ds(self):
        filters = ["not is_train"]
        return self.feature_manager.extract_dataframe(
            features_to_select=self.features_and_label, filters=filters
        )

    def get_test_ds(self):
        return self.get_val_ds()

    def get_submission_ds(self):
        return self.get_val_ds()
