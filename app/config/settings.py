"""Application settings and environment loading logic.

This module reads configuration from environment variables and validates the
values required for the RAG pipeline to run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for database access and Gemini integration."""

    gemini_api_key: str
    db_host: str = "127.0.0.1"
    db_port: int = 5432
    db_name: str = "ai_workshop"
    db_user: str = "postgres"
    db_password: str = "postgres"
    app_env: str = "development"
    gemini_text_model: str = "gemini-3.5-flash-lite"

    def validate(self) -> None:
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is missing. Create a .env file with your API key.")


def get_settings() -> Settings:
    settings = Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        db_host=os.getenv("DB_HOST", "127.0.0.1"),
        db_port=int(os.getenv("DB_PORT", "5432")),
        db_name=os.getenv("DB_NAME", "ai_workshop"),
        db_user=os.getenv("DB_USER", "postgres"),
        db_password=os.getenv("DB_PASSWORD", "postgres"),
        app_env=os.getenv("APP_ENV", "development"),
        gemini_text_model=os.getenv("GEMINI_TEXT_MODEL", "gemini-3.5-flash-lite"),
    )
    settings.validate()
    return settings
