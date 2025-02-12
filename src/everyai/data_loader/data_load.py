import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from everyai.data_loader.filter import default_filter
from everyai.utils.everyai_path import DATA_PATH
from everyai.utils.load_args import set_attrs_2class


class Data_loader:
    def __init__(
        self,
        data_filter: Callable[[pd.DataFrame], pd.DataFrame] = None,
        **data_kwargs,
    ):
        default_args = [
            "data_name",
            "question_column",
            "answer_column",
            "file_path",
            "data_type",
            "data_filter",
            "load_config",
            "max_count",
            "language",
        ]
        self.data_filter = data_filter
        set_attrs_2class(self, data_kwargs, default_args, default_args)
        if self.file_path is None:
            self.file_path = DATA_PATH / self.data_name

    def load_data(
        self, max_count: int = None, return_type: str = "list"
    ) -> list[dict] | pd.DataFrame:
        if Path(self.file_path).exists() or self.data_type == "huggingface":
            logging.info("Loading data from %s", self.file_path)
            match self.data_type:
                case "csv":
                    loaded_data = pd.read_csv(self.file_path)
                case "xlsx":
                    loaded_data = pd.read_excel(self.file_path)
                case "jsonl":
                    loaded_data = pd.read_json(self.file_path, lines=True)
                case "json":
                    loaded_data = pd.read_json(self.file_path)
                case "huggingface":
                    loaded_data = load_dataset(
                        self.file_path, **self.load_config
                    ).to_pandas()
                case _:
                    logging.error("Invalid file format")
        elif Path(self.file_path).exists():
            logging.error("Invalid file type")
        else:
            logging.error("File not found: %s", self.file_path)
        if self.data_filter is not None:
            loaded_data = self.data_filter(loaded_data)
            logging.info(
                "Default filter is applied, if you want to apply custom filter, "
                "please provide the filter function"
            )
        if max_count is not None and loaded_data is not None:
            loaded_data = loaded_data.head(max_count)
        else:
            logging.info("Max count is None and all the records will be loaded")
        loaded_data.rename(
            columns={
                self.question_column: "question",
                self.answer_column: "answer",
            },
            inplace=True,
        )
        if return_type == "pandas":
            result = loaded_data
        elif return_type == "list":
            result = loaded_data[["question", "answer"]].to_dict(orient="records")
        else:
            logging.error("Invalid return type")
        return result

    def apply_filter(self, orginal_data: pd.DataFrame) -> pd.DataFrame:
        return (
            orginal_data
            if self.data_filter is None
            else orginal_data[orginal_data.apply(self.data_filter, axis=1)]
        )
