from flask import Blueprint
from src.controllers.search import search
from src.controllers.default import default

# API V4, main blueprint to be registered with application
apiV1 = Blueprint('apiV1', __name__)

apiV1.register_blueprint(default, url_prefix="/v1")
apiV1.register_blueprint(search, url_prefix="/v1")
