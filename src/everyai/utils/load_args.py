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


def set_attrs_2class(self, classify_config:dict, allowed_keys:list, default_keys:list):
    """Set attributes for classification.

    Sets attributes based on the provided configuration, default values, and necessary keys.
    Logs warnings for invalid or missing keys.

    Args:
        classify_config (dict): Configuration dictionary.
        default_keys (list): Default values for attributes.
        necessary_keys (list): List of necessary keys.
    """
    for key, value in classify_config.items():
        if key in allowed_keys:
            setattr(self, key, value)
        else:
            logging.warning("Invalid key: %s", key)
    for key in default_keys:
        if key not in classify_config:
            setattr(self, key, None)
            logging.warning("Default key not provided: %s", key)
        
