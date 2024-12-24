# Description: This file contains the function to get the connection to the MongoDB database.
from pymongo import MongoClient


def get_mongo_connection(connection_string, database_name):
    client = MongoClient(connection_string)
    return client[database_name]
