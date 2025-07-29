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
    CAMERA_GROUP_SIZE,
    PROCESS_TIMEOUT,
    MIN_FRAMES_PER_GROUP,
    MAX_GROUP_WAIT,
    STALE_SEC,
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


# def async_publish(message):
#     """Kirim pesan secara asinkron"""
#     threading.Thread(
#         target=rabbitmq_client.publish, args=(message,), daemon=True
#     ).start()


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

        # start_time = time.time()

        results = model(frame, conf=DETECTION_THRESHOLD, stream=True)
        for result in results:
            # if result.boxes is not None:
            # logger.info(
            #     f"Detected from camera {camera_index} {len(result.boxes)} objects."
            # )
            # cv2.putText(
            #     frame,
            #     f"Detected: {len(result.boxes)} objects",
            #     (10, 120),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.9,
            #     (0, 255, 0),
            #     2,
            # )
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

                    # async_publish(
                    #     json.dumps(
                    #         {
                    #             "label": name,
                    #             "dist": float(distances[0][0]),
                    #             "location": camera_list[camera_index]["name"],
                    #             "camera_url": camera_list[camera_index]["url"],
                    #             "time": time.time(),
                    #             "Time+8": (
                    #                 datetime.datetime.now(datetime.timezone.utc)
                    #                 + datetime.timedelta(hours=8)
                    #             ).isoformat(),
                    #             # "blob_image": blob_base64,
                    #         }
                    #     )
                    # )

        # end_time = time.time()
        # processing_time = end_time - start_time

        # logger.info(
        #     f'[Camera {camera_list[camera_index]["name"]}] FPS: {fps:.2f} | Processing time: {processing_time:.2f}'
        # )
        processed_queues[camera_index].put({"frame": frame, "time": frame_data["time"]})


def process_batch_frames(start_index, end_index):
    batch_frames = []
    frame_infos = []

    for cam_id in range(start_index, end_index):
        try:
            frame_data = frame_queues[cam_id].get_nowait()  # NON-blocking
            batch_frames.append(frame_data["frame"])
            frame_infos.append((cam_id, frame_data["time"]))
        except queue.Empty:
            log_and_print(
                f"[Batch] Camera {cam_id} has no frame (skipped)", level="debug"
            )
            continue

    if not batch_frames:
        return  # Semua kamera kosong

    start_time = time.time()
    results = model(batch_frames, conf=DETECTION_THRESHOLD)

    for i, result in enumerate(results):
        if i >= len(frame_infos):
            log_and_print(
                "[Batch] Mismatch between results and frame_infos, skipping",
                level="error",
            )
            break
        cam_id, timestamp = frame_infos[i]
        frame = batch_frames[i]

        print(
            f"[Batch] Processing camera {cam_id} at {timestamp.isoformat()} with {len(result.boxes)} detections."
        )

        if result.boxes is not None:
            log_and_print(
                f"Detected from camera {cam_id} {len(result.boxes)} objects.",
                level="info",
            )
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
                if int(box.cls) != 0:
                    continue

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

                log_and_print(
                    f"[Batch] Camera {cam_id}: {name} (dist: {distances[0][0]:.2f})",
                    level="info",
                )

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

        end_time = time.time()
        processing_time = end_time - start_time
        fps = 1 / processing_time if processing_time > 0 else 0

        log_and_print(
            f'[Camera {camera_list[cam_id]["name"]}] FPS: {fps:.2f} | Processing time: {processing_time:.2f}',
            level="info",
        )
        processed_queues[cam_id].put({"frame": frame, "time": timestamp})


def process_camera(camera_id, camera_info):
    log_and_print(
        f"Starting camera thread {camera_id}: {camera_info['name']} from {camera_info['url']}",
        level="info",
    )

    while True:
        try:
            log_and_print(f"Trying to connect to Camera {camera_id}...")
            start_time_ntp = get_ntp_time()

            container = av.open(
                camera_info["url"],
                options={
                    "rtsp_transport": "tcp",
                    "stimeout": "5000000",
                    "max_delay": "5000000",
                    "rtsp_flags": "prefer_tcp",
                    "reorder_queue_size": "0",
                },
            )

            log_and_print(f"Camera {camera_id} connected successfully.", level="info")
            video_stream = next(
                (s for s in container.streams if s.type == "video"), None
            )
            if not video_stream:
                raise Exception("No video stream found")

            time_base = video_stream.time_base
            frame_count = 0
            prev_pts = -1
            stuck_counter = 0

            for packet in container.demux(video=0):
                for frame_data in packet.decode():
                    if frame_data.pts == prev_pts:
                        stuck_counter += 1
                    else:
                        stuck_counter = 0
                        prev_pts = frame_data.pts

                    if stuck_counter > 50:
                        log_and_print(
                            f"[Camera {camera_id}] WARNING: PTS stagnan (>50x), mungkin feed stuck",
                            level="warning",
                        )
                    try:
                        frame = frame_data.to_ndarray(format="bgr24")
                        frame_time = calculate_current_time(
                            start_time_ntp, frame_data.pts, time_base
                        )

                        frame_count += 1
                        if frame_queues[camera_id].empty():
                            try:
                                frame_queues[camera_id].put_nowait(
                                    {"frame": frame, "time": frame_time}
                                )
                                with ts_lock:  # ⬅️ Tambahan baru
                                    last_frame_ts[camera_id] = time.time()
                            except queue.Full:
                                log_and_print(
                                    f"Queue full for camera {camera_id}, dropping frame.",
                                    level="warning",
                                )
                    except Exception as e:
                        log_and_print(
                            f"Decode error for camera {camera_id}: {e}", level="warning"
                        )
                        continue

        except Exception as e:
            log_and_print(
                f"[Camera {camera_id}] Error: {e}. Retrying in {RETRY_DELAY}s...",
                level="error",
            )
            time.sleep(RETRY_DELAY)

        finally:
            with contextlib.suppress(Exception):
                if "container" in locals():
                    container.close()
                    log_and_print(f"Camera {camera_id} stream closed.", level="info")


def rotate_per_batch_group():
    global current_group
    total_groups = (num_cameras + CAMERA_GROUP_SIZE - 1) // CAMERA_GROUP_SIZE

    while True:
        start = current_group * CAMERA_GROUP_SIZE
        end = min(start + CAMERA_GROUP_SIZE, num_cameras)

        log_and_print(
            f"[Rotator-Batch] Activating group {current_group + 1}/{total_groups}: cameras {start} to {end - 1}",
            level="info",
        )

        # Kosongkan frame lama
        for i in range(start, end):
            while not frame_queues[i].empty():
                try:
                    frame_queues[i].get_nowait()
                except queue.Empty:
                    break

        rot_start_time = time.time()

        # Tunggu hingga ada minimal 1 frame dari grup ini
        wait_start = time.time()
        wait_timeout = 1.5  # detik
        has_fresh_frame = False
        while time.time() - wait_start < wait_timeout:
            with ts_lock:
                now = time.time()
                for i in range(start, end):
                    if not frame_queues[i].empty() and (now - last_frame_ts[i]) <= 1.0:
                        has_fresh_frame = True
                        break
            if not has_fresh_frame:
                log_and_print(
                    f"[Rotator-Batch] Group {current_group} skipped (no fresh frames)",
                    level="warning",
                )
                with group_lock:
                    current_group = (current_group + 1) % total_groups
                time.sleep(0.2)
                continue  # Langsung ke grup selanjutnya
            time.sleep(0.05)

        if has_fresh_frame:
            process_batch_frames(start, end)

        rot_elapsed = time.time() - rot_start_time
        log_and_print(
            f"[Rotator-Batch] Group {current_group} processed in {rot_elapsed:.2f} seconds",
            level="info",
        )

        with group_lock:
            current_group = (current_group + 1) % total_groups

        time.sleep(0.2)  # short pause sebelum next group


if __name__ == "__main__":
    logger = logging.getLogger("cctv_ml_inference")
    log_and_print = Logger(logger).log_and_print
    try:
        status_data = load_json(PREDICT_STATUS_PATH)

        # Load camera list
        with open(SOURCE_URLS_PATH, "r") as f:
            camera_list = json.load(f)
        num_cameras = len(camera_list)
        last_frame_ts = [0] * num_cameras  # global timestamp list
        ts_lock = threading.Lock()
        log_and_print(f"Number of cameras: {num_cameras}", level="info")

        current_group = 0
        group_lock = threading.Lock()  # ⬅️ Tambahkan ini

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
        # rabbitmq_client = RabbitMQClient(host=RABBITMQ_HOST, queue_name=QUEUE_NAME)

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

        start_thread(rotate_per_batch_group)
        # Display frames
        while True:
            active_group = (current_group - 1) % (
                (num_cameras + CAMERA_GROUP_SIZE - 1) // CAMERA_GROUP_SIZE
            )
            start_idx = active_group * CAMERA_GROUP_SIZE
            end_idx = min(start_idx + CAMERA_GROUP_SIZE, num_cameras)

            for i in range(start_idx, end_idx):
                try:
                    frame_data = processed_queues[i].get(timeout=1)
                except queue.Empty:
                    continue

                cv2.putText(
                    frame_data["frame"],
                    f"{camera_list[i]['name']} : Batch {active_group + 1}",
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

                cv2.imshow(f'{camera_list[i]["name"]} + {i}', frame_data["frame"])

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Cleanup
        for i in range(num_cameras):
            frame_queues[i].put(None)
        for thread in threads:
            thread.join()
        cv2.destroyAllWindows()
        # rabbitmq_client.close()
        status_data.update({"status": "completed"})
        save_json(PREDICT_STATUS_PATH, {"status": "completed"})
        logger.info("Prediction completed")
    except Exception as e:
        print(f"Prediction failed: {e}")
        status_data.update({"status": "failed", "error": str(e)})
        save_json(PREDICT_STATUS_PATH, {"status": "failed", "error": str(e)})
        logger.error(f"Prediction failed: {e}")
