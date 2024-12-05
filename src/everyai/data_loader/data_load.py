import logging
from pathlib import Path

import pandas as pd
from datasets import load_dataset


class Data_loader:
    def __init__(
        self,
        data_name: str,
        question_column: str="question",
        answer_column:str="answer",
        file_path: str | Path = None,
        data_type: str = None,
    ):
        self.data_name = data_name
        if file_path is None:
            self.file_name_or_path = data_name
        else:
            self.file_name_or_path = file_path
        if data_type is None:
            self.file_type = Path(self.file_name_or_path).suffix
        else:
            self.file_type = data_type
        self.question_column = question_column
        self.answer_column = answer_column

    def load_data2list(self, max_count: int = None):
        data = None
        if Path(self.file_name_or_path).exists() or self.file_type == "huggingface":
            match self.file_type:
                case "csv":
                    data = pd.read_csv(self.file_name_or_path)
                case "xlsx":
                    data = pd.read_excel(self.file_name_or_path)
                case "jsonl":
                    data = pd.read_json(self.file_name_or_path, lines=True)
                case "json":
                    data = pd.read_json(self.file_name_or_path)
                case "huggingface":
                    data = load_dataset(path=self.file_name_or_path)["train"].to_pandas()
                case _:
                    logging.error("Invalid file format")
        else:
            logging.error(f"File not found: {self.file_name_or_path}" or "Invalid file type")
        
        if max_count is not None and data is not None:
            data = data.head(max_count)
        else:
            logging.info('Max count is None and all the records will be loaded')
        data.rename(columns={self.question_column: "question", self.answer_column: "answer"}, inplace=True)
        return data[["question","answer"]].to_dict(orient="records")

if __name__ == "__main__":
    # loader = Data_loader("test.csv", "question")
    # data = loader.load_data()
    # print(data)
    # loader = Data_loader("test.xlsx", "question")
    # data = loader.load_data()
    # print(data)
    # loader = Data_loader("test.jsonl", "question")
    # data = loader.load_data()
    # print(data)
    # loader = Data_loader("test.json", "question")
    # data = loader.load_data()
    # print(data)
    loader = Data_loader("wanghw/human-ai-comparison", "question")
    data = loader.load_data2list()
    print(data)
    loader = Data_loader("test.invalid", "question")
    data = loader.load_data2list()
    print(data)
