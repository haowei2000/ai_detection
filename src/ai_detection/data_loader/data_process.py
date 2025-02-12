import logging
import re
import string
from pathlib import Path

import jieba

from ai_detection.utils.everyai_path import (EN_STOP_WORD_PATH,
                                             ZH_STOP_WORD_PATH)


def remove_punctuation(text: str) -> str:
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

    return re.sub(f"[{chinese_punc}]", "", text)


def load_stopwords(file_path: str | Path) -> set[str]:
    """
    Load stopwords from a text file.

    Args:
        file_path (str): Path to the stopwords file

    Returns:
        set: Set of stopwords
    """
    if file_path not in [EN_STOP_WORD_PATH, ZH_STOP_WORD_PATH]:
        raise ValueError("Invalid stopwords file path")
    with open(file_path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f}


def remove_stopwords(text: str, lang="both", stopwords: str | Path | set | list = None):
    """
    Remove stopwords from text in English and/or Chinese.

    Args:
        text (str): Input text containing stopwords
        lang (str): Language selection ('en', 'zh', or 'both')
        stopwords (str | Path | set | list, optional): Custom stopwords to remove. Defaults to None.

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
        match lang.lower():
            case ["en", "english"]:
                stopwords = en_stopwords
            case ["zh", "chinese"]:
                stopwords = zh_stopwords
            case _:
                stopwords = en_stopwords.union(zh_stopwords)
    words = text.split(" ")
    words = [word for word in words if word not in stopwords]
    return " ".join(words)


def chinese_split(text) -> str:
    """
    Split Chinese text into words.

    Args:
        text (str): Input Chinese text

    Returns:
        text: string of Chinese words with 1 space split
    """
    return " ".join(jieba.lcut(text))


def split_remove_stopwords_punctuation(text: str, language="both") -> str:
    """
    Split Chinese text into words, remove punctuation, and remove stopwords.

    Args:
        text (str): Input text containing words, punctuation, and stopwords.
        language (str): Language selection ('en', 'zh', or 'both').
        Defaults to 'both'.

    Returns:
        str: Processed text with words split, punctuation removed,
        and stopwords removed.
    """
    if language.lower() in ["zh", "chinese"]:
        text = chinese_split(text)
    else:
        text = text.lower()
    text = remove_punctuation(text)
    text = remove_stopwords(text, lang=language)
    text = text.replace("\n", "")
    text = text.strip()
    return text
