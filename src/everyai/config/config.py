from pathlib import Path

import yaml

from everyai.everyai_path import (BERT_TOPIC_CONFIG_PATH, CLASSIFY_CONFIG_PATH,
                                  DATA_LOAD_CONFIG_PATH, GENERATE_CONFIG_PATH,
                                  MONGO_CONFIG_PATH)


def get_config(file_path: Path):
    if file_path not in [
        GENERATE_CONFIG_PATH,
        DATA_LOAD_CONFIG_PATH,
        CLASSIFY_CONFIG_PATH,
        MONGO_CONFIG_PATH,
        BERT_TOPIC_CONFIG_PATH,
    ]:
        raise ValueError("Invalid config file path")
    with open(file_path, "r") as file:
        return yaml.safe_load(file)
