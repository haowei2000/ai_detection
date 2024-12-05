from ast import match_case
import logging
from math import log
from pathlib import Path
import pymongo.database
from pymongo.mongo_client import MongoClient
import pymongo
import pandas as pd


class EveryaiDataset:
    def __init__(
        self,
        dataname: str,
        datas: pd.DataFrame = None,
        ai_list: list[str] = None,
        language: str = "English",
    ):
        self.data_name: str = dataname
        if ai_list is not None:
            self.ai_list = ai_list
        else:
            ai_list = []
        if datas is not None:
            self.datas: pd.DataFrame = datas
        else:
            self.datas: pd.DataFrame = pd.DataFrame(
                columns=["question", "human", *ai_list]
            )
        self.language = language
        self.max_length = 0
        self.min_length = 0

    def add_ai(self, ai_name: str):
        self.ai_list.append(ai_name)
        self.datas.columns.append(ai_name)
        logging.info(f"Add model: {ai_name}")

    def get_records_with_1ai(self, ai_name: str):
        return self.datas[["question", "human", ai_name]].to_records()

    def insert_ai_response(self, question, ai_name: str, ai_response: str):
        if ai_name not in self.datas.columns:
            self.add_ai(ai_name)
        else:
            logging.info(f"AI {ai_name} exists in the dataset")
        if self.datas[self.datas["question"] == question].empty:
            logging.info(f"Inserting new question: {question}")
            new_row = pd.DataFrame({"question": [question], ai_name: [ai_response]})
            self.datas = pd.concat(
                [self.datas, new_row],
                ignore_index=True,
            )
        else:
            self.datas.loc[self.datas["question"] == question, ai_name] = (
                ai_response
            )

    def insert_human_response(self, question, human_response: str):
        if self.datas[self.datas["question"] == question].empty:
            logging.info(f"Inserting new question: {question}")
            new_row = pd.DataFrame({"question": [question], "human": [human_response]})
            self.datas = pd.concat(
                [self.datas, new_row],
                ignore_index=True,
            )
        else:
            self.datas.loc[self.datas["question"] == question, "human"] = human_response

    def output_question(self):  # -> Iterator:
        return iter(self.datas["question"])

    def _save2mongodb(self, database: pymongo.database.Database):
        logging.info(f"Saving dataset to mongodb: {database}")
        collection = database[self.data_name]
        self.datas["timestamp"] = pd.Timestamp.now()
        collection.insert_many(self.datas.to_dict(orient="records"))

    def save(self, path_or_database: str | Path = None, format: str = "csv"):
        if format == "mongodb":
            self._save2mongodb(path_or_database)
        else:
            if path_or_database is None:
                path_or_database = f"{self.data_name}_dataset.{format}"
            else:
                logging.info(f"Save dataset to {path_or_database}")
            if isinstance(path_or_database, str):
                path_or_database = Path(path_or_database)
            else:
                logging.error(f"Invalid file name: {path_or_database}")
            if path_or_database.suffix != f".{format}":
                logging.warning(f"Change file format to {format}")
                path_or_database = path_or_database.with_suffix(f".{format}")
            else:
                logging.info(f"Saving dataset to {path_or_database}")
            match format:
                case "csv":
                    self.datas.to_csv(path_or_database, index=False)
                case "xlsx":
                    self.datas.to_excel(path_or_database, index=False)
                case "json":
                    self.datas.to_json(path_or_database, orient="records")
                case _:
                    logging.error(f"Invalid format: {format}")
