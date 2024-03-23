from flask import Blueprint

default = Blueprint('default', __name__)

@default.route('')
def index():
    return "Welcome to my Service"