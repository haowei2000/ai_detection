import logging
from pathlib import Path
from typing import Callable

import pandas as pd
from datasets import load_dataset

from everyai.data_loader.filter import default_filter
from everyai.everyai_path import DATA_PATH


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

    def load_data2list(self, max_count: int = None):
        if (
            Path(self.file_name_or_path).exists()
            or self.file_type == "huggingface"
        ):
            match self.file_type:
                case "csv":
                    dataset = pd.read_csv(self.file_name_or_path)
                case "xlsx":
                    dataset = pd.read_excel(self.file_name_or_path)
                case "jsonl":
                    dataset = pd.read_json(self.file_name_or_path, lines=True)
                case "json":
                    dataset = pd.read_json(self.file_name_or_path)
                case "huggingface":
                    dataset = load_dataset(path=self.file_name_or_path)[
                        "train"
                    ].to_pandas()
                case _:
                    logging.error("Invalid file format")
        elif Path(self.file_name_or_path).exists():
            logging.error("Invalid file type")
        else:
            logging.error(f"File not found: {self.file_name_or_path}")
        if self.filter is not None:
            dataset = self.filter(dataset)
        else:
            dataset = default_filter(dataset)
            logging.info(
                (
                    "Default filter is applied, if you want to apply custom filter, "
                    "please provide the filter function"
                )
            )
        if max_count is not None and dataset is not None:
            dataset = dataset.head(max_count)
        else:
            logging.info(
                "Max count is None and all the records will be loaded"
            )
        dataset.rename(
            columns={
                self.question_column: "question",
                self.answer_column: "answer",
            },
            inplace=True,
        )
        return dataset[["question", "answer"]].to_dict(orient="records")

    def apply_filter(self, orginal_data: pd.DataFrame) -> pd.DataFrame:
        return (
            orginal_data
            if self.filter is None
            else orginal_data[orginal_data.apply(self.filter, axis=1)]
        )


if __name__ == "__main__":
    loader = Data_loader("wanghw/human-ai-comparison", "question")
    data = loader.load_data2list()
    print(data)
    loader = Data_loader("test.invalid", "question")
    data = loader.load_data2list()
    print(data)
