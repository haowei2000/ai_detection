import logging

import pandas as pd
import torch
from tqdm import tqdm

from everyai.classifier.fusion_classifer import PLClassifer
from everyai.classifier.huggingface_classifier import HuggingfaceClassifer
from everyai.classifier.sklearn_classifier import SklearnClassifer
from everyai.data_loader.data_load import Data_loader
from everyai.data_loader.data_process import split_remove_stopwords_punctuation
from everyai.data_loader.everyai_dataset import EveryaiDataset
from everyai.explanation.explain import LimeExplanation, ShapExplanation
from everyai.generator.generate import Generator
from everyai.topic.my_bertopic import create_topic
from everyai.utils.everyai_path import (BERT_TOPIC_CONFIG_PATH,
                                        CLASSIFY_CONFIG_PATH,
                                        DATA_LOAD_CONFIG_PATH, DATA_PATH,
                                        FIG_PATH, GENERATE_CONFIG_PATH)
from everyai.utils.load_config import get_config


def generate():
    logging.basicConfig(level=logging.WARNING)
    generate_list_configs = get_config(GENERATE_CONFIG_PATH)
    logging.info("Generate configs: %s", generate_list_configs)
    data_list_configs = get_config(file_path=DATA_LOAD_CONFIG_PATH)
    logging.info("Data configs: %s", data_list_configs)
    for data_config in data_list_configs["data_list"]:
        data_loader = Data_loader(
            **data_config,
        )
        max_count = data_config["max_count"] if "max_count" in data_config else None
        qa_datas = data_loader.load_data(max_count=max_count)
        everyai_dataset = EveryaiDataset(
            datas=pd.DataFrame(qa_datas),
            dataname=data_config["data_name"],
        )
        everyai_dataset.save(
            path_or_database=DATA_PATH / everyai_dataset.data_name,
            file_format="csv",
        )
        for generate_config in generate_list_configs["generate_list"]:
            logging.info("Generate config: %s", generate_config)
            generator = Generator(config=generate_config)
            for data in tqdm(qa_datas, desc="Generating data", total=len(qa_datas)):
                if everyai_dataset.record_exist(
                    data["question"], generate_config["model_name"]
                ):
                    logging.info("Record exists: %s", data["question"])
                else:
                    ai_response: str = generator.generate(data["question"])
                    everyai_dataset.upsert2mongo(
                        data["question"],
                        generate_config["model_name"],
                        ai_response,
                    )
                    everyai_dataset.upsert2mongo(
                        data["question"],
                        "human",
                        data["answer"],
                    )
                    everyai_dataset.insert_ai_response(
                        question=data["question"],
                        ai_name=generate_config["model_name"],
                        ai_response=ai_response,
                    )
                    everyai_dataset.insert_human_response(
                        question=data["question"],
                        human_response=data["answer"],
                    )
            torch.cuda.empty_cache()
            # mongodb_config = get_config(file_path=MONGO_CONFIG_PATH)
            # db = get_mongo_connection(**mongodb_config)
            # everyai_dataset.save(path_or_database=db, file_format="mongodb")
            everyai_dataset.save()


def topic():
    logging.basicConfig(level=logging.WARNING)
    data_list_configs = get_config(file_path=DATA_LOAD_CONFIG_PATH)
    logging.info("Data config: %s", data_list_configs)
    for data_config in data_list_configs["data_list"]:
        everyai_dataset = EveryaiDataset(
            dataname=data_config["data_name"],
            language=data_config["language"],
        )
        everyai_dataset.read(file_format="mongodb")
        logging.info("Loaded data: %s", everyai_dataset.data_name)
        topic_config = get_config(file_path=BERT_TOPIC_CONFIG_PATH)
        for catogeory in everyai_dataset.ai_list + ["human"]:
            logging.info("Category: %s", catogeory)
            docs = everyai_dataset.datas[catogeory].tolist()
            logging.info("Number of documents: %d", len(docs))
            new_docs = [
                split_remove_stopwords_punctuation(
                    doc, language=everyai_dataset.language
                )
                for doc in docs
            ]
            create_topic(
                docs=new_docs,
                output_folder=FIG_PATH / everyai_dataset.data_name / catogeory,
                topic_config=topic_config,
            )
            logging.info("Topic created for %s", catogeory)


def classify():
    logging.basicConfig(level=logging.INFO)
    data_list_configs = get_config(file_path=DATA_LOAD_CONFIG_PATH)
    logging.info("Data config: %s", data_list_configs)
    for data_config in data_list_configs["data_list"]:
        everyai_dataset = EveryaiDataset(
            dataname=data_config["data_name"],
            language=data_config["language"],
        )
        everyai_dataset.read(file_format="mongodb")
        texts, labels = everyai_dataset.get_records(only2class=True)
        logging.info("Label: %s", set(labels))
        for classify_config in get_config(file_path=CLASSIFY_CONFIG_PATH)[
            "classifier_list"
        ]:
            match classify_config["classifier_type"]:
                case "sklearn":
                    text_classifier = SklearnClassifer(
                        **classify_config,
                    )
                case "huggingface":
                    text_classifier = HuggingfaceClassifer(**classify_config)
                case "pl":
                    text_classifier = PLClassifer(
                        **classify_config,
                    )
                case _:
                    raise ValueError("Classifier type not supported")
            text_classifier.load_data(
                texts, labels, data_name=everyai_dataset.data_name
            )
            text_classifier.process_data()
            text_classifier.train()
            text_classifier.test()
            text_classifier.show_score()
            logging.info("Model saved for %s", classify_config["model_name"])
            lime_explanation = LimeExplanation(classifier=text_classifier)
            lime_explanation.explain()
            shap_explanation = ShapExplanation(classifier=text_classifier)
            shap_explanation.explain()


def main():
    logging.basicConfig(level=logging.INFO)
    generate()
    topic()
    classify()


if __name__ == "__main__":
    main()
