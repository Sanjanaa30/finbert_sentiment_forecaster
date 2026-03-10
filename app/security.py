import os
from pathlib import Path

from fastapi import Header, HTTPException, status
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def get_api_key() -> str | None:
    return os.getenv("API_KEY")

def verify_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    api_key = get_api_key()
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server API key is not configured.",
        )
    if not x_api_key or x_api_key != api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
