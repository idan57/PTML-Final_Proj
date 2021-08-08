import sys
from pathlib import Path

from flask_app.app import MainApp
import logging
import numpy as np

MIN_RAND = 1000
MAX_RAND = 1000000


def set_logger():
    logs_folder = Path("logs")

    if not logs_folder.is_dir():
        logs_folder.mkdir()
    log_file = str(logs_folder / f"example_{np.random.randint(MIN_RAND, MAX_RAND)}.log")
    logging.basicConfig(format="[%(levelname)s] [%(asctime)s] [%(name)s]: %(message)s",
                        handlers=[logging.FileHandler(log_file),
                                  logging.StreamHandler()],
                        level=logging.DEBUG)


if __name__ == '__main__':
    # start the main flask app execution
    set_logger()
    logging.info("================== STARTING APP ==================")
    MainApp().run()
