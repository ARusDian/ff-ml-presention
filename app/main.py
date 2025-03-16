from fastapi import FastAPI, Depends
from app.routers import camera, user, train, predict, logs
from app.middleware.auth import validate_api_key
from app.loggers.app_logger import app_logger

app = FastAPI()

# Register routers with API Token authentication
app.include_router(
    camera.router,
    prefix="/camera",
    tags=["Camera"],
    dependencies=[Depends(validate_api_key)],
)
app.include_router(
    user.router, prefix="/user", tags=["User"], dependencies=[Depends(validate_api_key)]
)
app.include_router(
    train.router,
    prefix="/train",
    tags=["Training"],
    dependencies=[Depends(validate_api_key)],
)
app.include_router(
    predict.router,
    prefix="/predict",
    tags=["Prediction"],
    dependencies=[Depends(validate_api_key)],
)
app.include_router(
    logs.router,
    prefix="/logs",
    tags=["Logs"],
    dependencies=[Depends(validate_api_key)],
)

@app.get("/")
def read_root():
    app_logger.info("Root endpoint accessed")
    return {"message": "Welcome to the CCTV Machine Learning API!"}
