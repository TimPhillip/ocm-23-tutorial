import logging
import os
import shutil


def setup_logging(logfile=None):

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler()
        ]
    )


def make_temp_dir(path="./temp"):

    if os.path.exists(path):
        shutil.rmtree(path)

    assert not os.path.exists(path)

    os.mkdir(path)
