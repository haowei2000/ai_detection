import pandas as pd

from everyai.data_loader.mongo_connection import get_mongo_connection

if __name__ == "__main__":
    database = get_mongo_connection(None, "everyai")
    collection = database["reddit_finance_43_250k"]
    # Load data from MongoDB collection into a DataFrame
    data = pd.DataFrame(list(collection.find()))
    print(data["question"].nunique())
    result = data.groupby("question", as_index=False).last()
    print(result.shape)
    result.to_csv("output.csv", index=False)
