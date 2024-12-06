import logging
import re
import string
from pathlib import Path

import jieba

from everyai.everyai_path import EN_STOP_WORD_PATH, ZH_STOP_WORD_PATH


def remove_punctuation(text):
    """
    Remove both English and Chinese punctuation marks from the text.

    Args:
        text (str): Input text containing punctuation marks

    Returns:
        str: Text with punctuation marks removed
    """
    # Define Chinese punctuation marks
    chinese_punc = (
        "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—''‛"
        "„‟…‧﹏"
    )

    # Create translation table for English punctuation
    translator = str.maketrans("", "", string.punctuation)

    # Remove English punctuation
    text = text.translate(translator)

    # Remove Chinese punctuation using regex
    text = re.sub(f"[{chinese_punc}]", "", text)

    return text


def load_stopwords(file_path: str | Path) -> set[str]:
    """
    Load stopwords from a text file.

    Args:
        file_path (str): Path to the stopwords file

    Returns:
        set: Set of stopwords
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f)


def remove_stopwords(
    text: str, lang="both", stopwords: str | Path | set | list = None
):
    """
    Remove stopwords from text in English and/or Chinese.

    Args:
        text (str): Input text containing stopwords
        lang (str): Language selection ('en', 'zh', or 'both')

    Returns:
        str: Text with stopwords removed
    """
    # English stopwords
    en_stopwords = load_stopwords(EN_STOP_WORD_PATH)
    zh_stopwords = load_stopwords(ZH_STOP_WORD_PATH)
    if isinstance(stopwords, (str, Path)):
        stopwords = load_stopwords(stopwords)
    elif isinstance(stopwords, (set, list)):
        stopwords = set(stopwords)
    else:
        if lang == "English" or lang == "en":
            stopwords = en_stopwords
            logging.info("Using English stopwords")
        elif lang == "zh" or lang == "Chinese":
            stopwords = zh_stopwords
            logging.info("Using Chinese stopwords")
        else:  # both
            stopwords = en_stopwords.union(zh_stopwords)
            logging.info("Using both English and Chinese stopwords")
    words = text.split(" ")
    words = [
        word
        for word in words
        if word not in stopwords and word.lower() not in stopwords
    ]
    return " ".join(words)


def chinese_split(text):
    """
    Split Chinese text into words.

    Args:
        text (str): Input Chinese text

    Returns:
        text: string of Chinese words with 1 space split
    """
    return " ".join(jieba.lcut(text))


def split_remove_stopwords_punctuation(text:str, language="English") -> str:
    text = chinese_split(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text, lang=language)
    text = text.strip()
    return text