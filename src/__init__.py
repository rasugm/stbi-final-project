from dotenv import load_dotenv
from flask import Flask

from config.stage import Config
from src.routes import apiV1

from flask_cors import CORS

# loading environment variables
load_dotenv()

# declaring flask application
app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# calling the dev configuration
config = Config().development_config

# making our application to use dev env
app.env = config.ENV

# setting json sort keys to false
app.json.sort_keys = False

app.register_blueprint(apiV1)

# import api blueprint to register it with app
# app.register_blueprint(apiV1, url_prefix='/v1')
