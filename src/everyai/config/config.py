from pathlib import Path

import yaml


def get_config(file_path: Path) -> dict:
    with open(file_path, "r") as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as exc:
            print(f"Error reading YAML file: {exc}")
