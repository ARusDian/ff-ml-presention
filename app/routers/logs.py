from fastapi import APIRouter, HTTPException
import os
from fastapi.responses import FileResponse
from app.loggers.app_logger import app_logger as logger, LOG_DIR as APP_LOG_DIR
from app.loggers.inference_logger import LOG_DIR as INFERENCE_LOG_DIR
from app.loggers.train_logger import LOG_DIR as TRAIN_LOG_DIR

router = APIRouter()


LOG_DIRECTORIES = {
    "app": APP_LOG_DIR,
    "inference": INFERENCE_LOG_DIR,
    "train": TRAIN_LOG_DIR,
}

# Ensure log directories exist
for log_dir in LOG_DIRECTORIES.values():
    os.makedirs(log_dir, exist_ok=True)


@router.get("/{log_type}/logs")
def list_logs(log_type: str):
    """List all log files for a specific log type."""
    if log_type not in LOG_DIRECTORIES:
        logger.error(f"Invalid log type: {log_type}")
        raise HTTPException(status_code=400, detail="Invalid log type")

    log_dir = LOG_DIRECTORIES[log_type]
    try:
        files = os.listdir(log_dir)
        log_files = [f for f in files if f.endswith(".log")]
        logger.info(f"Listed log files for {log_type}: {log_files}")
        return {"log_files": log_files}
    except Exception as e:
        logger.error(f"Failed to list log files for {log_type}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{log_type}/logs/{log_file}")
def get_log(log_type: str, log_file: str):
    """Get a specific log file for a specific log type."""
    if log_type not in LOG_DIRECTORIES:
        logger.error(f"Invalid log type: {log_type}")
        raise HTTPException(status_code=400, detail="Invalid log type")

    log_dir = LOG_DIRECTORIES[log_type]
    log_path = os.path.join(log_dir, log_file)
    if not os.path.exists(log_path):
        logger.error(f"Log file not found: {log_path}")
        raise HTTPException(status_code=404, detail="Log file not found")
    logger.info(f"Retrieved log file: {log_path}")
    return FileResponse(log_path)


@router.delete("/{log_type}/logs/{log_file}")
def delete_log(log_type: str, log_file: str):
    """Delete a specific log file for a specific log type."""
    if log_type not in LOG_DIRECTORIES:
        logger.error(f"Invalid log type: {log_type}")
        raise HTTPException(status_code=400, detail="Invalid log type")

    log_dir = LOG_DIRECTORIES[log_type]
    log_path = os.path.join(log_dir, log_file)
    if not os.path.exists(log_path):
        logger.error(f"Log file not found: {log_path}")
        raise HTTPException(status_code=404, detail="Log file not found")
    try:
        os.remove(log_path)
        logger.info(f"Deleted log file: {log_path}")
        return {"message": "Log file deleted"}
    except Exception as e:
        logger.error(f"Failed to delete log file: {log_path}, error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
