from collections.abc import Callable
import logging
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from everyai.data_loader.filter import default_filter
from everyai.utils.everyai_path import DATA_PATH


class Data_loader:
    def __init__(
        self,
        data_name: str,
        question_column: str = "question",
        answer_column: str = "answer",
        file_path: str | Path = None,
        data_type: str = None,
        data_filter: Callable[[pd.DataFrame], pd.DataFrame] = None,
    ):
        self.data_name = data_name
        if file_path is None:
            self.file_name_or_path = DATA_PATH / data_name
        else:
            self.file_name_or_path = file_path
        if data_type is None:
            self.file_type = Path(self.file_name_or_path).suffix
        else:
            self.file_type = data_type
        self.question_column = question_column
        self.answer_column = answer_column
        self.filter = data_filter

    def load_data(
        self, max_count: int = None, return_type: str = "list"
    ) -> list[dict] | pd.DataFrame:
        if (
            Path(self.file_name_or_path).exists()
            or self.file_type == "huggingface"
        ):
            logging.info("Loading data from %s", self.file_name_or_path)
            match self.file_type:
                case "csv":
                    loaded_data = pd.read_csv(self.file_name_or_path)
                case "xlsx":
                    loaded_data = pd.read_excel(self.file_name_or_path)
                case "jsonl":
                    loaded_data = pd.read_json(
                        self.file_name_or_path, lines=True
                    )
                case "json":
                    loaded_data = pd.read_json(self.file_name_or_path)
                case "huggingface":
                    loaded_data = load_dataset(path=self.file_name_or_path)[
                        "train"
                    ].to_pandas()
                case _:
                    logging.error("Invalid file format")
        elif Path(self.file_name_or_path).exists():
            logging.error("Invalid file type")
        else:
            logging.error("File not found: %s", self.file_name_or_path)
        if self.filter is not None:
            loaded_data = self.filter(loaded_data)
        else:
            loaded_data = default_filter(loaded_data)
            logging.info(
                "Default filter is applied, if you want to apply custom filter, "
                "please provide the filter function"
            )
        if max_count is not None and loaded_data is not None:
            loaded_data = loaded_data.head(max_count)
        else:
            logging.info(
                "Max count is None and all the records will be loaded"
            )
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
            result = loaded_data[["question", "answer"]].to_dict(
                orient="records"
            )
        else:
            logging.error("Invalid return type")
        return result

    def apply_filter(self, orginal_data: pd.DataFrame) -> pd.DataFrame:
        return (
            orginal_data
            if self.filter is None
            else orginal_data[orginal_data.apply(self.filter, axis=1)]
        )


if __name__ == "__main__":
    loader = Data_loader("wanghw/human-ai-comparison", "question")
    data = loader.load_data()
    print(data)
    loader = Data_loader("test.invalid", "question")
    data = loader.load_data()
    print(data)
