import os
import pickle
import json
import numpy as np
from tqdm import tqdm
import face_recognition
import faiss
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.loggers.train_logger import train_logger as logger

# Constants
PICKLE_FILE = "machine_learning/models/face_data.pkl"
KNOWN_FACES_DIR = "machine_learning/db"
TRAIN_STATUS_FILE = "machine_learning/config/train_status.json"


# Utilities
def update_train_status(status, progress=0):
    """Update training status in train_status.json."""
    status_data = {"status": status, "progress": progress}
    try:
        with open(TRAIN_STATUS_FILE, "w") as f:
            json.dump(status_data, f, indent=4)
        logger.info(f"Training status updated: {status_data}")
    except Exception as e:
        logger.error(f"Failed to update training status: {e}")


# Main Training Process
def train_model():
    known_face_encodings, known_face_names = [], []
    update_train_status("training", 0)
    logger.info("Starting training process...")

    try:
        # Gather all users and their images
        user_folders = [
            folder
            for folder in os.listdir(KNOWN_FACES_DIR)
            if os.path.isdir(os.path.join(KNOWN_FACES_DIR, folder))
        ]
        total_images = sum(
            len(os.listdir(os.path.join(KNOWN_FACES_DIR, folder)))
            for folder in user_folders
        )

        if total_images == 0:
            logger.warning("No images found for training.")
            update_train_status("completed", 100)
            return {"message": "No images found for training."}

        processed_images = 0
        # Process each user folder
        for folder in tqdm(user_folders, desc="Processing Users"):
            user_face = []
            folder_path = os.path.join(KNOWN_FACES_DIR, folder)
            for file in os.listdir(folder_path):
                image_path = os.path.join(folder_path, file)
                logger.info(f"Processing file: {image_path}")

                try:
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        user_face.append(encodings[0])
                        logger.info(f"Successfully encoded: {image_path}")
                    else:
                        logger.warning(f"No face found in image: {image_path}")
                except Exception as e:
                    logger.error(f"Error processing file {image_path}: {e}")

                # Update progress
                processed_images += 1
                progress = int((processed_images / total_images) * 100)
                update_train_status("training", progress)

            if user_face:
                known_face_encodings.append(user_face)
                known_face_names.append(folder)
                logger.info(f"Encodings for user '{folder}' added successfully")

        # Flatten encodings and build FAISS index
        if known_face_encodings:
            flattened_encodings = [e for faces in known_face_encodings for e in faces]
            dimension = len(flattened_encodings[0])
            index = faiss.IndexFlatL2(dimension)
            known_face_names_map = []

            for i, name in enumerate(known_face_names):
                for encoding in known_face_encodings[i]:
                    index.add(np.array([encoding], dtype=np.float32))
                    known_face_names_map.append(name)

            # Save data to pickle
            data = {
                "known_face_encodings": known_face_encodings,
                "known_face_names": known_face_names,
                "known_face_names_map": known_face_names_map,
                "faiss_index": index.reconstruct_n(0, index.ntotal),
            }
            with open(PICKLE_FILE, "wb") as f:
                pickle.dump(data, f)

            logger.info(f"Training completed successfully. Data saved to {PICKLE_FILE}")
            update_train_status("completed", 100)
        else:
            logger.warning(
                "No encodings were generated. Training process completed with no data."
            )
            update_train_status("completed", 100)

    except Exception as e:
        logger.error(f"Unexpected error during training: {e}")
        update_train_status("error", 0)
        raise


if __name__ == "__main__":
    train_model()
