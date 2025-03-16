from fastapi import APIRouter, HTTPException
from typing import List, Dict
from app.utils import load_json, save_json
from app.loggers.app_logger import app_logger as logger
from machine_learning.config.inference_config import SOURCE_URLS_PATH, VIDEO_DIR
import os

router = APIRouter()

@router.get("/")
def get_camera():
    try:
        data = load_json(SOURCE_URLS_PATH)
        logger.info("Fetched camera URLs successfully")
        return data
    except FileNotFoundError:
        logger.error("Camera URLs file not found")
        raise HTTPException(status_code=404, detail="Camera URLs file not found")

@router.post("/")
def update_camera(new_urls: List[Dict[str, str]]):
    try:
        # Save only the relative paths
        for item in new_urls:
            if "url" in item:
                item["url"] = os.path.relpath(item["url"], VIDEO_DIR)
        save_json(SOURCE_URLS_PATH, new_urls)
        logger.info(f"Updated camera URLs: {new_urls}")
        return {"message": "Camera URLs updated successfully"}
    except Exception as e:
        logger.error(f"Failed to update camera URLs: {e}")
        raise HTTPException(status_code=500, detail="Failed to update camera URLs")
