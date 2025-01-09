import logging

classfiy_allowed_keys = [
    "model_name",
    "tokenizer_name",
    "classifier_type",
    "split_size",
    "train_args",
    "tokenizer_config",
    "pipeline",
]


def set_attrs_2class(self, classify_config, allowed_keys, necessary_keys):
    for key, value in classify_config.items():
        if key in allowed_keys:
            setattr(self, key, value)
        else:
            logging.warning("Invalid key: %s", key)
    for key in necessary_keys:
        if key not in classify_config:
            setattr(self, key, None)
            logging.warning("Necessary key not provided: %s", key)
