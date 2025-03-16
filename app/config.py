import os
from dotenv import load_dotenv
from app.loggers.app_logger import app_logger

# Ensure .env file exists before loading
CWD = os.getcwd()
ENV_PATH = os.path.join(CWD, ".env")

if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)
else:
    app_logger.error("Missing .env file")

# Debugging: Ensure API_KEYS can be read
API_KEYS = os.getenv("API_KEYS")

if not API_KEYS:
    API_KEYS = []
else:
    API_KEYS = [key.strip() for key in API_KEYS.split(",")]

# Fix BASE_DIR to avoid extra "app/" level
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MACHINE_LEARNING_DIR = os.path.join(BASE_DIR, "machine_learning")

# File Paths (Ensure directories exist before use)
CAMERA_URLS_PATH = os.path.join(MACHINE_LEARNING_DIR, "config/video_data.json")
DB_PATH = os.path.join(MACHINE_LEARNING_DIR, "db")
TRAIN_SCRIPT = os.path.join(MACHINE_LEARNING_DIR, "scripts", "train.py")
TRAIN_STATUS_PATH = os.path.join(MACHINE_LEARNING_DIR, "config", "train_status.json")
