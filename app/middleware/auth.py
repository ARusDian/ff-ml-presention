from fastapi.security.api_key import APIKeyHeader
from fastapi import HTTPException, Security
from app.config import API_KEYS

api_key_header = APIKeyHeader(name="X-API-Key")

def validate_api_key(api_key: str = Security(api_key_header)):
    if api_key not in API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key
