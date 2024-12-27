import logging
from pathlib import Path

import pandas as pd
import pymongo
import pymongo.database

from everyai.config.config import get_config
from everyai.data_loader.mongo_connection import get_mongo_connection
from everyai.everyai_path import DATA_PATH, MONGO_CONFIG_PATH


class EveryaiDataset:
    def __init__(
        self,
        dataname: str,
        datas: pd.DataFrame = None,
        ai_list: list[str] = None,
        language: str = "English",
    ):
        self.data_name: str = dataname
        self.ai_list = ai_list if ai_list is not None else []
        if datas is not None:
            self.datas: pd.DataFrame = datas
        else:
            self.datas: pd.DataFrame = pd.DataFrame(columns=["question", "human"])
            if ai_list is not None:
                for ai_name in ai_list:
                    self.datas[ai_name] = None
            else:
                logging.warning("No AI list provided")
        self.language: str = language
        self.max_length: int = 0
        self.min_length: int = 0

    def add_ai(self, ai_name: str):
        self.ai_list.append(ai_name)
        self.datas[ai_name] = None
        logging.info("Add model: %s", ai_name)

    def get_records_with_1ai(self, ai_list: list[str] = None):
        texts = []
        labels = []
        if ai_list is None:
            ai_list = self.ai_list
        for label in ["human"] + ai_list:
            texts.extend(self.datas[label])
            labels.extend([label] * len(self.datas[label]))
        return texts, labels

    def insert_ai_response(self, question, ai_name: str, ai_response: str):
        if ai_name not in self.datas.columns:
            self.add_ai(ai_name)
        else:
            logging.info("AI %s exists in the dataset", ai_name)
        question_exists = self.datas[self.datas["question"] == question].empty
        if question_exists:
            self._update_new_row(question, ai_name, ai_response)
        else:
            self.datas.loc[self.datas["question"] == question, ai_name] = ai_response

    def insert_human_response(self, question, human_response: str):
        if self.datas[self.datas["question"] == question].empty:
            self._update_new_row(question, "human", human_response)
        else:
            self.datas.loc[self.datas["question"] == question, "human"] = human_response

    # TODO Rename this here and in `insert_ai_response` and `insert_human_response`
    def _update_new_row(self, question, arg1, arg2):
        logging.info("Inserting new question: %s", question)
        new_row = pd.DataFrame({"question": [question], arg1: [arg2]})
        self.datas = pd.concat([self.datas, new_row], ignore_index=True)

    def output_question(self):  # -> Iterator:
        return iter(self.datas["question"])

    def _save2mongodb(self, database: pymongo.database.Database):
        logging.info("Saving dataset to mongodb: %s", database)
        collection = database[self.data_name]
        if "timestamp" not in self.datas.columns:
            self.datas["timestamp"] = pd.Timestamp.now()
        else:
            self.datas.loc[self.datas["timestamp"].isnull(), "timestamp"] = (
                pd.Timestamp.now()
            )
        collection.insert_many(self.datas.to_dict(orient="records"))

    def _load_from_mongodb(self, database: pymongo.database.Database):
        logging.info("Loading dataset from mongodb: %s", database)
        collection = database[self.data_name]
        data = pd.DataFrame(list(collection.find()))
        data = data.drop(columns=["_id"], errors="ignore")
        data = data.sort_values(by="timestamp", ascending=False)
        data = data.drop_duplicates(subset=["question"], keep="first")
        data = data.drop(columns=["timestamp"])
        self.datas = data

    def load(self, path_or_database: str | Path = None, formatter: str = "csv"):
        if formatter == "mongodb":
            if path_or_database is None:
                path_or_database = self._initialize_mongo_connection()
            else:
                logging.info("Load dataset from %s", path_or_database)
            self._load_from_mongodb(path_or_database)
        else:
            if path_or_database is None:
                path_or_database = DATA_PATH / f"{self.data_name}.{formatter}"
            else:
                logging.info("Load dataset from %s", path_or_database)
            if isinstance(path_or_database, str):
                path_or_database = Path(path_or_database)
            if not isinstance(path_or_database, Path):
                logging.error("Invalid file name: %s", path_or_database)
            if path_or_database is not None and path_or_database.suffix != f".{formatter}":
                logging.warning("Change file format to %s", formatter)
                path_or_database = path_or_database.with_suffix(f".{formatter}")
            else:
                logging.info("Loading dataset from %s", path_or_database)
            match formatter:
                case "csv":
                    self.datas = pd.read_csv(path_or_database)
                case "xlsx":
                    self.datas = pd.read_excel(path_or_database)
                case "json":
                    self.datas = pd.read_json(path_or_database)
                case _:
                    logging.error("Invalid format: %s", formatter)
        if self.datas is not None:
            self.ai_list = list(
                set(self.datas.columns) - {"question", "human", "timestamp"}
            )

    def save(self, path_or_database: str | Path = None, formatter: str = "csv"):
        if formatter == "mongodb":
            if path_or_database is None:
                path_or_database = self._initialize_mongo_connection()
            else:
                logging.info("Save dataset to %s", path_or_database)
            self._save2mongodb(path_or_database)
        else:
            if path_or_database is None:
                path_or_database = f"{self.data_name}.{formatter}"
            else:
                logging.info("Save dataset to %s", path_or_database)
            if isinstance(path_or_database, str):
                path_or_database = Path(path_or_database)
            if not isinstance(path_or_database, Path):
                logging.error("Invalid file name: %s", path_or_database)
                return
            if path_or_database.suffix != f".{formatter}":
                logging.warning("Change file format to %s", formatter)
                path_or_database = path_or_database.with_suffix(f".{formatter}")
            else:
                logging.info("Saving dataset to %s", path_or_database)
            match formatter:
                case "csv":
                    self.datas.to_csv(path_or_database, index=False)
                case "xlsx":
                    self.datas.to_excel(path_or_database, index=False)
                case "json":
                    self.datas.to_json(path_or_database, orient="records")
                case _:
                    logging.error("Invalid format: %s", formatter)

    @staticmethod
    def _initialize_mongo_connection():# -> Database:
        mongodb_config = get_config(MONGO_CONFIG_PATH)
        result = get_mongo_connection(**mongodb_config)
        logging.info("Use default mongodb: %s", result)
        return result
