from pymongo import MongoClient

class Mongo:
    def __init__(self):
        self.client = MongoClient('mongodb://root:admin123%23@localhost:27017/?authMechanism=SCRAM-SHA-1&authSource=admin')
        self.db = self.client['kpu']
        self.collection = self.db['tbl_caleg']