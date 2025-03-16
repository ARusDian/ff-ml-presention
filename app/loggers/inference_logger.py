import logging
import os

LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "logs",
    "inference",
)

os.makedirs(LOG_DIR, exist_ok=True)

INFERENCE_LOG_FILE = os.path.join(LOG_DIR, "inference.log")
DEBUG_LOG_FILE = os.path.join(LOG_DIR, "debug.log")
ERROR_LOG_FILE = os.path.join(LOG_DIR, "errors.log")
WARNING_LOG_FILE = os.path.join(LOG_DIR, "warning.log")

# Configure inference logger
inference_logger = logging.getLogger("cctv_ml_inference")
inference_logger.setLevel(logging.DEBUG)

# Info handler
info_handler = logging.FileHandler(INFERENCE_LOG_FILE)
info_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
info_handler.setLevel(logging.INFO)

# Debug handler
debug_handler = logging.FileHandler(DEBUG_LOG_FILE)
debug_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
debug_handler.setLevel(logging.DEBUG)

# Error handler
error_handler = logging.FileHandler(ERROR_LOG_FILE)
error_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
error_handler.setLevel(logging.ERROR)

# Warning handler
warning_handler = logging.FileHandler(WARNING_LOG_FILE)
warning_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
warning_handler.setLevel(logging.WARNING)

inference_logger.addHandler(info_handler)
inference_logger.addHandler(debug_handler)
inference_logger.addHandler(error_handler)
inference_logger.addHandler(warning_handler)
