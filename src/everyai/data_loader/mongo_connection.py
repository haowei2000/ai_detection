# Description: This file contains the function to get the connection to the MongoDB database.
import logging
import os

from pymongo import MongoClient


def get_mongo_connection(connection_string:str, database_name:str):# -> Database:
    if connection_string is None or not connection_string:
        logging.info("Use the connection string in environment variable")
        connection_string = os.getenv("MONGO_CONNECTION_STRING")
    client = MongoClient(connection_string)
    return client[database_name]
