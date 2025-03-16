from fastapi import APIRouter
import subprocess
import psutil
import os
import time
from datetime import datetime, timedelta
from ..utils import load_json, save_json
from app.loggers.app_logger import app_logger as logger
from machine_learning.config.inference_config import PREDICT_SCRIPT, PREDICT_STATUS_PATH, PREDICT_PID_PATH

router = APIRouter()

def get_pid():
    """Helper function untuk membaca PID dari file"""
    try:
        with open(PREDICT_PID_PATH, "r") as f:
            return int(f.read().strip())
    except FileNotFoundError:
        return None


def is_process_running(pid):
    """Cek apakah proses dengan PID masih berjalan"""
    return psutil.pid_exists(pid)


def stop_process():
    """Hentikan proses inference.py jika berjalan"""
    pid = get_pid()
    if pid and is_process_running(pid):
        process = psutil.Process(pid)
        process.terminate()  # Bisa juga pakai process.kill() jika perlu
        os.remove(PREDICT_PID_PATH)  # Hapus file PID setelah proses dihentikan
        save_json(PREDICT_STATUS_PATH, {"status": "stopped"})
        logger.info(f"Stopped inference process with PID {pid}")
        return {"message": f"Prediction process {pid} stopped"}
    return {"message": "No running prediction process found"}


@router.post("/")
def start_prediction():
    """Starts inference.py as a background process."""
    existing_pid = get_pid()

    if existing_pid and is_process_running(existing_pid):
        return {"message": "Prediction process is already running", "pid": existing_pid}

    try:

        process = subprocess.Popen(
            ["python", PREDICT_SCRIPT],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,  # Prevents child process from being killed if API restarts
        )

        # Save PID for tracking
        with open(PREDICT_PID_PATH, "w") as f:
            f.write(str(process.pid))
            
        save_json(PREDICT_STATUS_PATH, {"status": "predicting", "pid": process.pid, "start_time": time.time()})


        logger.info(f"Started inference process with PID {process.pid}")
        return {"message": "Prediction started", "pid": process.pid}

    except Exception as e:
        logger.error(f"Failed to start prediction: {e}")
        save_json(PREDICT_STATUS_PATH, {"status": "failed", "error": str(e)})
        return {"message": f"Failed to start prediction: {e}"}


@router.get("/status")
def check_status():
    """Check if inference.py is still running."""
    try:
        status = load_json(PREDICT_STATUS_PATH)
        logger.info("Fetched prediction status")

        # Calculate runtime if the process is running
        if status.get("status") == "predicting" and "start_time" in status:
            start_time = status["start_time"]
            current_time = time.time()
            runtime = current_time - start_time
            status["runtime"] = str(timedelta(seconds=int(runtime)))

        return status
    except FileNotFoundError:
        logger.error("Prediction status file not found")
        return {"status": "unknown"}


@router.post("/stop")
def stop_prediction():
    """Endpoint untuk menghentikan inference.py"""
    return stop_process()
