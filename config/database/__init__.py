from config.database.mongo import Mongo


class Config:
    def __init__(self):
        self.development_config = Mongo()
