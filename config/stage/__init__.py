from config.stage.development import DevelopmentConfig


class Config:
    def __init__(self):
        self.development_config = DevelopmentConfig()
