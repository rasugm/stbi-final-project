from src import app, config
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    app.run(host=config.stage.HOST, port=config.stage.PORT, debug=config.stage.DEBUG)
