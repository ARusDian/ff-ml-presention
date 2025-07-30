import cv2
from ultralytics import YOLO
import face_recognition
import numpy as np
import os
import pandas as pd
from queue import Queue
from threading import Thread, Lock
import ntplib
import time
import faiss
import av
import pickle
import json
import pika
import threading
import queue
import datetime
import pytz
import logging
import contextlib
import sys
import base64

# Define the environment variable
ENV = os.getenv("ENV", "testing")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.utils import save_json, load_json
from app.loggers.inference_logger import inference_logger as logger
from machine_learning.config.inference_config import (
    PREDICT_STATUS_PATH,
    SOURCE_URLS_PATH,
    PICKLE_FILE,
    FAISS_INDEX_FILE,
    KNOWN_FACES_DIR,
    ENGINE_PATH,
    MODEL_PATH,
    PROCESS_EVERY_N_FRAMES,
    MATCH_THRESHOLD,
    DETECTION_THRESHOLD,
    QUEUE_NAME,
    RABBITMQ_HOST,
    MAX_QUEUE_SIZE,
    RETRY_DELAY,
    VIDEO_DIR,
)


class Logger:
    def __init__(self, logger):
        self.logger = logger

    def log_and_print(self, message, level="debug"):
        if level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        elif level == "debug":
            self.logger.debug(message)
        print(message)


class RabbitMQClient:
    def __init__(
        self, host="localhost", queue_name="hello", max_retries=5, retry_delay=3
    ):
        self.host = host
        self.queue_name = queue_name
        self.connection = None
        self.channel = None
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logging.getLogger("cctv_ml_inference")
        self.connect()

    def connect(self):
        for attempt in range(1, self.max_retries + 1):
            try:
                self.logger.info(f"[RabbitMQ] Connecting to {self.host}...")
                self.connection = pika.BlockingConnection(
                    pika.ConnectionParameters(
                        self.host,
                        heartbeat=600,
                        blocked_connection_timeout=300,
                        retry_delay=self.retry_delay,
                    )
                )
                self.channel = self.connection.channel()
                self.channel.queue_declare(queue=self.queue_name)
                self.logger.info(f"[RabbitMQ] Connected to {self.host}.")
                return
            except (
                pika.exceptions.AMQPConnectionError,
                pika.exceptions.ChannelError,
            ) as e:
                self.logger.warning(
                    f"[RabbitMQ] Connection failed: {e}. Retrying ({attempt}/{self.max_retries})..."
                )
                time.sleep(self.retry_delay)
            except Exception as e:
                self.logger.error(
                    f"[RabbitMQ] Unknown error: {e}. Retrying ({attempt}/{self.max_retries})..."
                )
                time.sleep(self.retry_delay)

        self.logger.error(
            f"[RabbitMQ] Failed to connect after {self.max_retries} attempts."
        )
        raise Exception(
            f"[RabbitMQ] Failed to connect after {self.max_retries} attempts."
        )

    def publish(self, message):
        retries = 0
        while retries < self.max_retries:
            try:
                if (
                    not self.connection
                    or self.connection.is_closed
                    or self.channel.is_closed
                ):
                    self.logger.info("[RabbitMQ] Reconnecting...")
                    self.connect()

                self.channel.basic_publish(
                    exchange="",
                    routing_key=self.queue_name,
                    body=message.encode("utf-8"),
                )
                # self.logger.info(f"[RabbitMQ] Published message: {message}")
                return True
            except (
                pika.exceptions.AMQPConnectionError,
                pika.exceptions.ChannelError,
            ) as e:
                self.logger.warning(
                    f"[RabbitMQ] Error publishing message: {e}. Retrying..."
                )
                retries += 1
                time.sleep(self.retry_delay)

        self.logger.error(
            f"[RabbitMQ] Failed to publish message after {self.max_retries} attempts."
        )
        return False

    def close(self):
        if self.connection and not self.connection.is_closed:
            self.logger.info("[RabbitMQ] Closing connection...")
            self.connection.close()


def get_ntp_time():
    ntp_servers = ["asia.pool.ntp.org"]
    for server in ntp_servers:
        try:
            client = ntplib.NTPClient()
            response = client.request(server, version=3)
            utc_time = datetime.datetime.fromtimestamp(response.tx_time)
            return utc_time.replace(tzinfo=pytz.utc)
        except Exception:
            # log_and_print(
            #     "Gagal mendapatkan waktu dari NTP, menggunakan waktu lokal.",
            #     level="warning",
            # )
            continue  # Coba server lain jika gagal
    return datetime.datetime.now().replace(tzinfo=pytz.utc)


def calculate_current_time(start, frame_pts, time_base):
    if frame_pts is not None:
        return start + datetime.timedelta(seconds=float(frame_pts * time_base))
    return start


def async_publish(message):
    """Kirim pesan secara asinkron"""
    threading.Thread(
        target=rabbitmq_client.publish, args=(message,), daemon=True
    ).start()


# Load or create face database
def load_or_create_face_database():
    logger = logging.getLogger("cctv_ml_inference")
    logger.info("Loading known faces...")
    logger.info(f"PICKLE_FILE: {PICKLE_FILE}, FAISS_INDEX_FILE: {FAISS_INDEX_FILE}")
    if os.path.exists(PICKLE_FILE) and os.path.exists(FAISS_INDEX_FILE):
        logger.info(
            f"{PICKLE_FILE} and {FAISS_INDEX_FILE} exist, loading with memory-mapped mode..."
        )

        # Load face names
        with open(PICKLE_FILE, "rb") as f:
            data = pickle.load(f)

        # Load FAISS index in memory-mapped mode
        index = faiss.read_index(FAISS_INDEX_FILE, faiss.IO_FLAG_MMAP)
        data["faiss_index"] = index
        return data

    logger.info(f"{PICKLE_FILE} not found, generating...")

    known_face_encodings = []
    known_face_names = []

    for folder in os.listdir(KNOWN_FACES_DIR):
        user_face = []
        for file in os.listdir(f"{KNOWN_FACES_DIR}/{folder}"):
            image_path = f"{KNOWN_FACES_DIR}/{folder}/{file}"
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                user_face.append(encodings[0])

        if user_face:
            known_face_encodings.extend(user_face)
            known_face_names.extend([folder] * len(user_face))

    if not known_face_encodings:
        logger.warning("No faces found.")
        return {"known_face_names": [], "faiss_index": None}

    known_face_encodings = np.array(known_face_encodings, dtype=np.float32)
    dimension = known_face_encodings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIDMap2(index)

    ids = np.arange(len(known_face_names))
    index.add_with_ids(known_face_encodings, ids)

    # Save FAISS index in memory-mapped mode
    faiss.write_index(index, FAISS_INDEX_FILE)
    data = {
        "known_face_names": known_face_names,
    }

    with open(PICKLE_FILE, "wb") as f:
        pickle.dump(data, f)

    data["faiss_index"] = index
    return data


# Frame processing function
def process_frames(camera_index):
    logger = logging.getLogger("cctv_ml_inference")
    while True:
        try:
            frame_data = frame_queues[camera_index].get_nowait()
            frame = frame_data["frame"]
            if frame is None:
                break
        except Exception as e:
            # logger.error(f"Error mengambil frame dari antrian: {e}")
            continue

        start_time = time.time()

        results = model(frame, conf=DETECTION_THRESHOLD, stream=True)
        for result in results:
            if result.boxes is not None:
                # logger.info(
                #     f"Detected from camera {camera_index} {len(result.boxes)} objects."
                # )
                cv2.putText(
                    frame,
                    f"Detected: {len(result.boxes)} objects",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )
            for box in result.boxes:
                if int(box.cls) == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    face = frame[y1:y2, x1:x2]

                    face = np.array(face, dtype=np.uint8)
                    face_encoding = face_recognition.face_encodings(face)

                    if not face_encoding:
                        continue

                    face_encoding = np.array([face_encoding[0]], dtype=np.float32)
                    distances, indices = index.search(face_encoding, k=1)

                    if distances[0][0] < MATCH_THRESHOLD:
                        name = data["known_face_names"][indices[0][0]]
                    else:
                        name = "Unknown"

                    # logger.info(
                    #     f"Name: {name}, distance: {distances[0][0]}"
                    # )

                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"{name} ({distances[0][0]:.2f})",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        color,
                        2,
                    )

                    # padded_face = frame[y1 - 10 : y2 + 10, x1 - 10 : x2 + 10]

                    # _, buffer = cv2.imencode(".jpg", face)  # Encode sebagai JPEG
                    # blob_base64 = base64.b64encode(buffer).decode(
                    #     "utf-8"
                    # )  # Konversi ke Base64

                    async_publish(
                        json.dumps(
                            {
                                "label": name,
                                "dist": float(distances[0][0]),
                                "location": camera_list[camera_index]["name"],
                                "camera_url": camera_list[camera_index]["url"],
                                "time": time.time(),
                                "Time+8": (
                                    datetime.datetime.now(datetime.timezone.utc)
                                    + datetime.timedelta(hours=8)
                                ).isoformat(),
                                # "blob_image": blob_base64,
                            }
                        )
                    )

        end_time = time.time()
        # processing_time = end_time - start_time

        # logger.info(
        #     f'[Camera {camera_list[camera_index]["name"]}] FPS: {fps:.2f} | Processing time: {processing_time:.2f}'
        # )
        processed_queues[camera_index].put({"frame": frame, "time": frame_data["time"]})


def process_camera(camera_id, camera_info):
    logger = logging.getLogger("cctv_ml_inference")
    # logger.info(
    #     f"Processing camera {camera_id}: {camera_info['name']} from {camera_info['url']}..."
    # )

    camera_url = (
        os.path.join(VIDEO_DIR, camera_info["url"])
        if ENV == "testing"
        else camera_info["url"]
    )

    def reconnect(attempt=1):
        delay = min(RETRY_DELAY * (2 ** (attempt - 1)), 60)  # Maksimal 60 detik
        logger.info(f"Reconnecting Camera {camera_id} in {delay} seconds...")
        threading.Timer(delay, process_camera, args=(camera_id, camera_info)).start()

    while True:
        try:
            start_time_ntp = get_ntp_time()  # Sinkronisasi ulang waktu setiap reconnect
            logger.info(f"Trying to connect to Camera {camera_id}...")

            container = av.open(
                camera_url,
                options={
                    "rtsp_transport": "tcp",
                    "stimeout": "5000000",
                    "max_delay": "5000000",
                    "rtsp_flags": "prefer_tcp",
                    "reorder_queue_size": "0",
                },
            )
            logger.info(f"Camera {camera_id} connected successfully.")

            video_stream = next(
                (s for s in container.streams if s.type == "video"), None
            )
            if not video_stream:
                logger.error(f"No video stream found for camera {camera_id}!")
                return  # Keluar dari fungsi jika tidak ada stream

            time_base = video_stream.time_base
            frame_count = 0

            while True:
                frame = None
                for packet in container.demux(video=0):
                    try:
                        for frame_data in packet.decode():
                            frame = frame_data.to_ndarray(format="bgr24")
                            frame_time = calculate_current_time(
                                start_time_ntp, frame_data.pts, time_base
                            )
                            break
                        if frame is not None:
                            break
                    except Exception as e:
                        logger.error(
                            f"Error decoding frame from Camera {camera_id}: {e}"
                        )
                        continue

                if frame is not None:
                    frame_count += 1
                    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                        try:
                            frame_queues[camera_id].put_nowait(
                                {"frame": frame, "time": frame_time}
                            )
                        except queue.Full:
                            logger.warning(
                                f"Queue full for camera {camera_id}, dropping frame."
                            )
                else:
                    logger.warning(
                        f"Frame is None for camera {camera_id}, reconnecting..."
                    )
                    break  # Keluar dari loop jika frame None

        except (av.error.EOFError, av.error.OSError) as e:
            logger.error(f"Error with camera {camera_id}: {e}. Reconnecting...")
        except Exception as e:
            logger.error(
                f"Unexpected error with camera {camera_id}: {e}. Reconnecting..."
            )
        finally:
            with contextlib.suppress(Exception):
                container.close()
                logger.info(f"Camera {camera_id} stream closed.")

            reconnect()


if __name__ == "__main__":
    logger = logging.getLogger("cctv_ml_inference")
    log_and_print = Logger(logger).log_and_print
    try:
        status_data = load_json(PREDICT_STATUS_PATH)

        # Load camera list
        with open(SOURCE_URLS_PATH, "r") as f:
            camera_list = json.load(f)
        num_cameras = len(camera_list)
        log_and_print(f"Number of cameras: {num_cameras}", level="info")

        if not os.path.exists(ENGINE_PATH):
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(
                    f"Model file {MODEL_PATH} not found. Cannot export."
                )

            logger.info(f"{ENGINE_PATH} not found. Exporting YOLO model to TensorRT...")
            try:
                status_data.update({"status": "exporting"})
                save_json(PREDICT_STATUS_PATH, status_data)
                model = YOLO(MODEL_PATH).export(format="engine", half=True)
                logger.info(f"Model exported successfully to {ENGINE_PATH}")

                status_data.update({"status": "exported"})
                save_json(PREDICT_STATUS_PATH, status_data)
            except Exception as e:
                logger.error(f"Failed to export model: {e}")
                raise RuntimeError("Model export failed. Check logs for details.")
        else:
            logger.info(f"Model {ENGINE_PATH} found. Using existing model.")

        model = YOLO(ENGINE_PATH, task="detect")
        # Initialize RabbitMQ Client
        status_data.update({"status": "connecting rabbitmq client"})
        save_json(PREDICT_STATUS_PATH, status_data)
        rabbitmq_client = RabbitMQClient(host=RABBITMQ_HOST, queue_name=QUEUE_NAME)

        # Load or create face database
        status_data.update({"status": "loading"})
        save_json(PREDICT_STATUS_PATH, status_data)
        data = load_or_create_face_database()
        log_and_print(f"Loaded names: {data['known_face_names']}", level="info")
        index = data["faiss_index"]

        status_data.update({"status": "predicting"})
        save_json(PREDICT_STATUS_PATH, status_data)

        # Initialize queues for processing frames
        frame_queues = [Queue(maxsize=MAX_QUEUE_SIZE) for _ in range(num_cameras)]
        processed_queues = [Queue(maxsize=MAX_QUEUE_SIZE) for _ in range(num_cameras)]

        # Start threads

        threads = []

        def start_thread(target, args=()):
            thread = Thread(target=target, args=args, daemon=True)
            thread.start()
            threads.append(thread)

        # Start processing threads
        for i in range(num_cameras):
            start_thread(process_frames, (i,))

        # Start camera threads
        for i, camera_info in enumerate(camera_list):
            start_thread(process_camera, (i, camera_info))

        # Display frames
        while True:
            for i, camera_info in enumerate(camera_list):
                try:
                    frame_data = processed_queues[i].get(
                        timeout=1
                    )  # Wait for 1s to avoid busy looping
                except queue.Empty:
                    continue  # Skip if no frame available

                cv2.putText(
                    frame_data["frame"],
                    camera_info["name"],
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

                frame_time_str = (
                    frame_data["time"]
                    .astimezone(pytz.timezone("Asia/Shanghai"))
                    .strftime("%Y-%m-%d %H:%M:%S")
                )

                cv2.putText(
                    frame_data["frame"],
                    frame_time_str,
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow(f'{camera_info["name"]} + {i}', frame_data["frame"])

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Cleanup
        for i in range(num_cameras):
            frame_queues[i].put(None)
        for thread in threads:
            thread.join()
        cv2.destroyAllWindows()
        rabbitmq_client.close()
        status_data.update({"status": "completed"})
        save_json(PREDICT_STATUS_PATH, {"status": "completed"})
        logger.info("Prediction completed")
    except Exception as e:
        print(f"Prediction failed: {e}")
        status_data.update({"status": "failed", "error": str(e)})
        save_json(PREDICT_STATUS_PATH, {"status": "failed", "error": str(e)})
        logger.error(f"Prediction failed: {e}")
