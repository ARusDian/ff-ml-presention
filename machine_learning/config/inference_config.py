import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define the base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Define the machine learning directory
MACHINE_LEARNING_DIR = os.path.join(BASE_DIR, "machine_learning")

# Define paths for various files
PICKLE_FILE = os.getenv(
    "PICKLE_FILE", os.path.join(MACHINE_LEARNING_DIR, "models", "face_data.pkl")
)
FAISS_INDEX_FILE = os.getenv(
    "FAISS_INDEX_FILE", os.path.join(MACHINE_LEARNING_DIR, "models", "face_index.index")
)
KNOWN_FACES_DIR = os.getenv("KNOWN_FACES_DIR", os.path.join(MACHINE_LEARNING_DIR, "db"))
PREDICT_STATUS_PATH = os.getenv(
    "PREDICT_STATUS_PATH",
    os.path.join(MACHINE_LEARNING_DIR, "config", "predict_status.json"),
)
CAMERA_URLS_PATH = os.getenv(
    "CAMERA_URLS_PATH", os.path.join(MACHINE_LEARNING_DIR, "config", "camera_data.json")
)
VIDEO_URLS_PATH = os.getenv(
    "VIDEO_URLS_PATH", os.path.join(MACHINE_LEARNING_DIR, "config", "video_data.json")
)
TRAIN_STATUS_PATH = os.getenv(
    "TRAIN_STATUS_PATH",
    os.path.join(MACHINE_LEARNING_DIR, "config", "train_status.json"),
)
PREDICT_SCRIPT = os.getenv(
    "PREDICT_SCRIPT", os.path.join(MACHINE_LEARNING_DIR, "scripts", "inference.py")
)
PREDICT_PID_PATH = os.getenv(
    "PREDICT_PID_PATH", os.path.join(MACHINE_LEARNING_DIR, "config", "predict_pid.txt")
)

# Define other configurations
PROCESS_EVERY_N_FRAMES = int(os.getenv("PROCESS_EVERY_N_FRAMES", 5))
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", 0.15))
DETECTION_THRESHOLD = float(os.getenv("DETECTION_THRESHOLD", 0.20))
QUEUE_NAME = os.getenv("QUEUE_NAME", "inference_queue")
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", 10))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 5))

# Model paths
ENGINE_PATH = os.getenv(
    "ENGINE_PATH", os.path.join(MACHINE_LEARNING_DIR, "models", "yolov11n-face.engine")
)
MODEL_PATH = os.getenv(
    "MODEL_PATH", os.path.join(MACHINE_LEARNING_DIR, "models", "yolov11n-face.pt")
)

# Video directory
VIDEO_DIR = os.getenv("VIDEO_DIR", os.path.join(MACHINE_LEARNING_DIR, "videos"))

# Determine which configuration to use based on the environment
ENV = os.getenv("ENV", "production")
SOURCE_URLS_PATH = VIDEO_URLS_PATH if ENV == "testing" else CAMERA_URLS_PATH
