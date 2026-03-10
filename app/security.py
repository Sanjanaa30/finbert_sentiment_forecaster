import os
from fastapi import Header, HTTPException, status

API_KEY = os.getenv("API_KEY")

def verify_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    if not API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server API key is not configured.",
        )
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
