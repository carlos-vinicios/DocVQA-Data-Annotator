import pymongo
import os

def generate_connection(collection_name: str):
    conn_str = "mongodb+srv://root_user:CNfjWrKJlIh9pnSj@docvqa.olokopd.mongodb.net/DOC_VQA?retryWrites=true&w=majority" #os.environ.get("MONGODB_URI")
    db_name = "DOC_VQA"#os.environ.get("MONGODB_DB_NAME")

    print(db_name)
    client = pymongo.MongoClient(
        conn_str,
        serverSelectionTimeoutMS=5000000
    )

    db = client[db_name]
    return db[collection_name]