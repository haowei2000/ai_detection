import logging
from pathlib import Path

import pandas as pd
from datasets import load_dataset


class Data_loader:
    def __init__(
        self, file_name_or_path, question_column
    ):
        self.file_name_or_path = file_name_or_path
        self.question_column = question_column

    def load_data(self):
        data = None
        if Path(self.file_name_or_path).exists():
            match Path(self.file_name_or_path).suffix:
                case "csv":
                    data = pd.read_csv(self.file_name_or_path)
                case "xlsx":
                    data = pd.read_excel(self.file_name_or_path)
                case "jsonl":
                    data = pd.read_json(self.file_name_or_path, lines=True)
                case "json":
                    data = pd.read_json(self.file_name_or_path)
                case _:
                    logging.error("Invalid file format")
        else:
            data = load_dataset(self.file_name_or_path)['train'].to_pandas()
        data= data[self.question_column].to_list()
        return data

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
    data = loader.load_data()
    print(data)
    loader = Data_loader("test.invalid", "question")
    data = loader.load_data()
    print(data)