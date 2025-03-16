# Face Recognition Web Application

This project is a face recognition web application built using FastAPI. It includes functionalities for managing cameras, users, training models, making predictions, and viewing logs.

## Table of Contents

- [Face Recognition Web Application](#face-recognition-web-application)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Steps](#steps)
  - [Usage](#usage)
  - [API Endpoints](#api-endpoints)
    - [Camera](#camera)
    - [User Management](#user-management)
    - [Training](#training)
    - [Prediction](#prediction)
    - [Logs](#logs)
  - [Logging](#logging)
  - [Configuration](#configuration)
  - [Folder Structure](#folder-structure)
  - [Contributing](#contributing)
  - [License](#license)

## Installation

### Prerequisites

- [Anaconda](https://www.anaconda.com/products/distribution)
- [CUDA](https://developer.nvidia.com/cuda-downloads) (for GPU support)

### Steps

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/FaceRecog.git
    cd FaceRecog/WebApp/apiApp
    ```

2. Run the installation script:

    ```sh
    ./install.sh
    ```

This script will:
- Download and install Anaconda.
- Create a new conda environment with Python 3.10.
- Install dependencies from `requirements.txt`.
- Install `torch`, `torchvision`, and `torchaudio` with CUDA support.
- Set the environment as the default in `.bashrc`.

## Usage

1. Activate the conda environment:

    ```sh
    conda activate myenv
    ```

2. Run the FastAPI application:

    ```sh
    uvicorn app.main:app --reload
    ```

3. Access the API documentation at [http://localhost:8000/docs](http://localhost:8000/docs).

## API Endpoints

### Camera

- **Get Camera URLs**
  - `GET /camera/`
  - Fetches the list of camera URLs.

- **Update Camera URLs**
  - `POST /camera/`
  - Updates the list of camera URLs.

### User Management

- **Add User**
  - `POST /user/add_user/{user_name}`
  - Adds a new user.

- **Remove User**
  - `DELETE /user/remove_user/{user_name}`
  - Removes an existing user.

- **Add Image**
  - `POST /user/add_image/{user_name}`
  - Adds images for a user.

- **Remove Image**
  - `DELETE /user/remove_image/{user_name}`
  - Removes images for a user.

### Training

- **Start Training**
  - `POST /train/`
  - Starts the training process.

- **Get Training Status**
  - `GET /train/status`
  - Fetches the training status.

### Prediction

- **Start Prediction**
  - `POST /predict/`
  - Starts the prediction process.

- **Get Prediction Status**
  - `GET /predict/status`
  - Fetches the prediction status.

- **Stop Prediction**
  - `POST /predict/stop`
  - Stops the prediction process.

### Logs

- **List Logs**
  - `GET /logs/{log_type}/logs`
  - Lists all log files for a specific log type.

- **Get Log**
  - `GET /logs/{log_type}/logs/{log_file}`
  - Fetches a specific log file.

- **Delete Log**
  - `DELETE /logs/{log_type}/logs/{log_file}`
  - Deletes a specific log file.

## Logging

Logs are stored in the `logs` directory with separate subdirectories for application, inference, and training logs.

## Configuration

Configuration settings are stored in the `.env` file and various configuration files in the `machine_learning/config` directory.

## Folder Structure

The project directory structure is as follows:

```
FaceRecog/
├── WebApp/
│   ├── apiApp/
│   │   ├── app/
│   │   ├── logs/
│   │   ├── machine_learning/
│   │   ├── .env
│   │   ├── install.sh
│   │   ├── requirements.txt
│   │   ├── postman_collection.json
│   │   ├── README.md
│   │   └── .gitignore
│   └── ...
└── ...
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.

---

This project is owned by **PT Sentra Teknologi**.
