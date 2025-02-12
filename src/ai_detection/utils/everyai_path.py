from pathlib import Path

GENERATE_CONFIG_PATH = Path(__file__).parents[1] / "config" / "generate.yaml"
DATA_LOAD_CONFIG_PATH = Path(__file__).parents[1] / "config" / "data.yaml"
MONGO_CONFIG_PATH = Path(__file__).parents[1] / "config" / "mongodb.yaml"
BERT_TOPIC_CONFIG_PATH = Path(__file__).parents[1] / "config" / "topic.yaml"
DATA_PATH = Path(__file__).parents[1] / "data"
FIG_PATH = Path(__file__).parents[1] / "fig"
EN_STOP_WORD_PATH = Path(__file__).parents[1] / "data_loader" / "en_stopwords.txt"
ZH_STOP_WORD_PATH = Path(__file__).parents[1] / "data_loader" / "zh_stopwords.txt"
CLASSIFY_CONFIG_PATH = Path(__file__).parents[1] / "config" / "classify.yaml"
MODEL_PATH = Path(__file__).parents[1] / "model"
RESULT_PATH = Path(__file__).parents[1] / "result"
