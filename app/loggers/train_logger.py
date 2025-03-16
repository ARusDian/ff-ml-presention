import logging
import os

LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "logs",
    "train",
)

os.makedirs(LOG_DIR, exist_ok=True)

TRAIN_LOG_FILE = os.path.join(LOG_DIR, "train.log")
print(TRAIN_LOG_FILE)
# Configure training logger
train_logger = logging.getLogger("cctv_ml_train")
train_logger.setLevel(logging.INFO)
train_handler = logging.FileHandler(TRAIN_LOG_FILE)
train_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
train_logger.addHandler(train_handler)
