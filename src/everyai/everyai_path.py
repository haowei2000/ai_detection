from pathlib import Path

GENERATE_CONFIG_PATH = Path(__file__).parent / "config" / "generate.yaml"
DATA_LOAD_CONFIG_PATH = Path(__file__).parent / "config" / "data.yaml"
MONGO_CONFIG_PATH = Path(__file__).parent / "config" / "mongodb.yaml"
BERT_TOPIC_CONFIG_PATH = Path(__file__).parent / "config" / "topic.yaml"
DATA_PATH = Path(__file__).parent / "data"
FIG_PATH = Path(__file__).parent / "fig"
EN_STOP_WORD_PATH = Path(__file__).parent / "data_loader" / "en_stopwords.txt"
ZH_STOP_WORD_PATH = Path(__file__).parent / "data_loader" / "zh_stopwords.txt"
CLASSIFY_CONFIG_PATH = Path(__file__).parent / "config" / "classify.yaml"
MODEL_PATH = Path(__file__).parent / "model"
RESULT_PATH = Path(__file__).parent / "result"
