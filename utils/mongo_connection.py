import pymongo
import os

def generate_connection(collection_name: str):
    conn_str = os.environ.get("MONGODB_URI")
    db_name = os.environ.get("MONGODB_DB_NAME")

    print(db_name)
    client = pymongo.MongoClient(
        conn_str,
        serverSelectionTimeoutMS=5000000
    )

    db = client[db_name]
    return db[collection_name]
