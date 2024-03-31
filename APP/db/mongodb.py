# APP/db/mongodb.py
# MongoDB 연결, 데이터를 조회, 삽입하는 함수 정의
from pymongo import MongoClient
import pandas as pd

class MongoDB:
    def __init__(self, url="mongodb://localhost:27017"):
        try:
            self.client = MongoClient(url)
        except Exception as e:
            print(f"MongoDB 연결 실패: {e}")
            raise e
    
    def get_collection(self, db_name, collection_name):
        db = self.client[db_name]
        collection = db[collection_name]
        return collection

    def find_all(self, db_name, collection_name, query={}):
        collection = self.get_collection(db_name, collection_name)
        documents = list(collection.find(query))
        return documents

    def insert_one(self, db_name, collection_name, document):
        collection = self.get_collection(db_name, collection_name)
        collection.insert_one(document)
        
    
    def df_from_collection(self, db_name, collection_name, query={}):
        documents = self.find_all(db_name, collection_name, query)
        return pd.DataFrame(documents)

    
