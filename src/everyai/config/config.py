from pathlib import Path

import yaml


def get_config(file_path: Path) -> dict:
    if not file_path.is_file():
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    if file_path.suffix != ".yaml":
        raise ValueError("Only YAML files are supported.")

    with open(file_path, "r") as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(f"Error reading YAML file: {exc}")
