import logging

from sympy import im

from everyai.config.config import get_config
from everyai.everyai_path import (
    DATA_LOAD_CONFIG_PATH,
    GENERATE_CONFIG_PATH,
    DATA_PATH,
    MONGO_CONFIG_PATH,
)
from everyai.generator.generate import Generator
from everyai.data_loader.data_load import Data_loader
from everyai.data_loader.everyai_dataset import EveryaiDataset
from everyai.data_loader.mongo_connection import get_mongo_connection
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def generate():
    generate_config = get_config(GENERATE_CONFIG_PATH)
    logging.info(f"Generate config: {generate_config}")
    data_config = get_config(DATA_LOAD_CONFIG_PATH)
    logging.info(f"Data config: {data_config}")
    for data_config in data_config["data_list"]:
        for generate_config in generate_config["generate_list"]:
            generator = Generator(config=generate_config)
            data_loader = Data_loader(
                data_name=data_config["data_name"],
                question_column=data_config["question_column"],
                answer_column=data_config["answer_column"],
                file_path=data_config["file_path"],
                data_type=data_config["data_type"],
            )
            qa_datas = data_loader.load_data2list(max_count=data_config["max_count"])
            everyai_dataset = EveryaiDataset(
                dataname=data_config["data_name"],
                ai_list=[generate_config["model_name"]],
            )
            for data in tqdm(
                qa_datas, desc="Generating data", total=len(qa_datas)
            ):
                logging.info(f"Generate data: {data["question"]}")
                ai_response: str = generator.generate(data["question"])
                everyai_dataset.insert_ai_response(
                    question=data["question"],
                    ai_name=generate_config["model_name"],
                    ai_response=ai_response,
                )
                everyai_dataset.insert_human_response(
                    question=data["question"], human_response=data["answer"]
                )
                logging.info(f"AI response: {ai_response}")
                logging.info(f"Human response: {data['answer']}")
                everyai_dataset.save(
                    path_or_database=DATA_PATH / everyai_dataset.data_name,
                    format="csv",
                )
                mongodb_config = get_config(MONGO_CONFIG_PATH)
                db = get_mongo_connection(**mongodb_config)
                everyai_dataset.save(path_or_database=db, format="mongodb")


if __name__ == "__main__":
    generate()
