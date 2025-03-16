import logging
import os

LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "logs",
    "app",
)
os.makedirs(LOG_DIR, exist_ok=True)

APP_LOG_FILE = os.path.join(LOG_DIR, "app.log")
ERROR_LOG_FILE = os.path.join(LOG_DIR, "error.log")

# Configure application logger
app_logger = logging.getLogger("cctv_ml_app")
app_logger.setLevel(logging.INFO)
app_handler = logging.FileHandler(APP_LOG_FILE)
app_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
app_logger.addHandler(app_handler)

# Configure error logger
error_logger = logging.getLogger("cctv_ml_error")
error_logger.setLevel(logging.ERROR)
error_handler = logging.FileHandler(ERROR_LOG_FILE)
error_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
error_logger.addHandler(error_handler)
