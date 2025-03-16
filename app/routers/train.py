from fastapi import APIRouter
import subprocess
from app.utils import load_json, save_json
from app.loggers.app_logger import app_logger as logger
from app.config import TRAIN_SCRIPT, TRAIN_STATUS_PATH

router = APIRouter()


@router.post("/")
def train_model():
    try:
        save_json(TRAIN_STATUS_PATH, {"status": "training"})
        subprocess.Popen(
            ["python", TRAIN_SCRIPT], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        logger.info("Started training process")
        return {"message": "Training started"}
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise


@router.get("/status")
def train_status():
    try:
        status = load_json(TRAIN_STATUS_PATH)
        logger.info("Fetched training status")
        return status
    except FileNotFoundError:
        logger.warning("Training status file not found")
        return {"status": "unknown", "progress": 0}
