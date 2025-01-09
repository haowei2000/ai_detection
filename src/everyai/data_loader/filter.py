import pandas as pd


def default_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Default filter to get only human data,
    If you want to filter the data based on some condition,
    please provide the filter function in the Data_loader class
    """
    return df if "label" not in df.columns else df[df["label"] != "ai"]
