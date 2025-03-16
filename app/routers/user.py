from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List
import os
import shutil
from app.config import DB_PATH, TRAIN_STATUS_PATH
from app.loggers.app_logger import app_logger as logger
import json


def update_train_status(status, progress=0):
    """Update training status in train_status.json."""
    status_data = {"status": status, "progress": progress}
    try:
        with open(TRAIN_STATUS_PATH, "w") as f:
            json.dump(status_data, f, indent=4)
        logger.info(f"Training status updated: {status_data}")
    except Exception as e:
        logger.error(f"Failed to update training status: {e}")


router = APIRouter()


@router.post("/add_image/{user_name}")
def add_image(user_name: str, files: List[UploadFile] = File(...)):
    user_folder = os.path.join(DB_PATH, user_name)
    os.makedirs(user_folder, exist_ok=True)
    for file in files:
        file_path = os.path.join(user_folder, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info(f"Added image {file.filename} to {user_folder}")

    # Memperbarui status pelatihan
    update_train_status(
        "images updated, need to retrain", 0
    )  # Menandakan pelatihan perlu dilakukan ulang
    return {"message": f"Images added to {user_name}"}


@router.delete("/remove_image/{user_name}")
def remove_image(user_name: str, file_names: List[str] = Form(...)):
    user_folder = os.path.join(DB_PATH, user_name)
    if not os.path.exists(user_folder):
        logger.error(f"User folder not found: {user_folder}")
        raise HTTPException(status_code=404, detail="User folder not found")

    for file_name in file_names:
        file_path = os.path.join(user_folder, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Removed image {file_name} from {user_folder}")
        else:
            logger.warning(f"File {file_name} not found in {user_folder}")
            raise HTTPException(status_code=404, detail=f"File {file_name} not found")

    # Memperbarui status pelatihan setelah gambar dihapus
    update_train_status(
        "images updated, need to retrain", 0
    )  # Menandakan pelatihan perlu dilakukan ulang
    return {"message": f"Files removed from {user_name}"}


@router.post("/add_user/{user_name}")
def add_user(user_name: str):
    user_folder = os.path.join(DB_PATH, user_name)
    if os.path.exists(user_folder):
        logger.error(f"User already exists: {user_name}")
        raise HTTPException(status_code=400, detail="User already exists")

    os.makedirs(user_folder)
    logger.info(f"Added new user: {user_name}")

    # Memperbarui status pelatihan setelah pengguna ditambahkan
    update_train_status(
        "images updated, need to retrain", 0
    )  # Menandakan pelatihan perlu dilakukan ulang
    return {"message": f"User {user_name} added"}


@router.delete("/remove_user/{user_name}")
def remove_user(user_name: str):
    user_folder = os.path.join(DB_PATH, user_name)
    if not os.path.exists(user_folder):
        logger.error(f"User folder not found: {user_folder}")
        raise HTTPException(status_code=404, detail="User folder not found")

    shutil.rmtree(user_folder)
    logger.info(f"Removed user: {user_name}")

    # Memperbarui status pelatihan setelah pengguna dihapus
    update_train_status(
        "images updated, need to retrain", 0
    )  # Menandakan pelatihan perlu dilakukan ulang
    return {"message": f"User {user_name} removed"}
