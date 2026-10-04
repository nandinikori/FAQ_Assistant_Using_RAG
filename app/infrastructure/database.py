"""PostgreSQL connection layer for the FAQ assistant.

This module centralizes database configuration and query execution so the rest
of the application can work with a consistent client interface.
"""

from __future__ import annotations

from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor

from app.config.settings import Settings


class DatabaseClient:
    """Thin wrapper around the PostgreSQL connection and common operations."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def connect(self):
        """Create a PostgreSQL connection.

        Some Windows environments resolve localhost differently from 127.0.0.1,
        so the client tries the configured host first and then a reliable loopback
        fallback before failing.
        """
        hosts = []
        configured_host = (self.settings.db_host or "").strip()
        if configured_host:
            hosts.append(configured_host)
        fallback_hosts = ["127.0.0.1", "localhost"]
        for host in fallback_hosts:
            if host not in hosts:
                hosts.append(host)

        last_error = None
        for host in hosts:
            try:
                return psycopg2.connect(
                    host=host,
                    port=self.settings.db_port,
                    dbname=self.settings.db_name,
                    user=self.settings.db_user,
                    password=self.settings.db_password,
                    cursor_factory=RealDictCursor,
                )
            except Exception as exc:  # pragma: no cover - failure path is operational
                last_error = exc

        raise last_error or RuntimeError("Unable to connect to PostgreSQL.")

    def ping(self) -> bool:
        # Fast health check for startup validation.
        try:
            with self.connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    row = cur.fetchone()
                    if not row:
                        return False
                    value = next(iter(row.values()))
                    return value == 1
        except Exception:
            return False

    def execute(self, query: str, params: tuple[Any, ...] | None = None):
        # Execute a query and return rows in a dictionary-friendly format.
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params or ())
                return cur.fetchall()
