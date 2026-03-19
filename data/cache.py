"""
SQLite cache layer — fetch log and metadata storage.

Tracks when data was last fetched to avoid redundant API calls.
"""
import sqlite3
import os
from datetime import datetime, timezone, timedelta


_DEFAULT_DB_PATH = "data/store/store.db"


def get_connection(db_path: str = _DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Get a SQLite connection, creating the database and tables if needed."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    _ensure_tables(conn)
    return conn


def _ensure_tables(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fetch_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fetch_type TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            rows_fetched INTEGER,
            tickers TEXT,
            errors TEXT,
            warnings TEXT
        )
    """)
    conn.commit()


def log_fetch(conn: sqlite3.Connection, fetch_type: str, metadata: dict):
    """Record a fetch event."""
    conn.execute(
        "INSERT INTO fetch_log (fetch_type, timestamp, rows_fetched, tickers, errors, warnings) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            fetch_type,
            metadata.get("fetch_timestamp", datetime.now(timezone.utc).isoformat()),
            metadata.get("rows", 0),
            ",".join(metadata.get("tickers", [])),
            ",".join(metadata.get("errors", [])),
            ",".join(metadata.get("warnings", [])),
        ),
    )
    conn.commit()


def last_fetch_time(conn: sqlite3.Connection, fetch_type: str = "daily") -> datetime | None:
    """Return the timestamp of the most recent fetch, or None if never fetched."""
    cursor = conn.execute(
        "SELECT timestamp FROM fetch_log WHERE fetch_type = ? ORDER BY id DESC LIMIT 1",
        (fetch_type,),
    )
    row = cursor.fetchone()
    if row:
        return datetime.fromisoformat(row[0])
    return None


def is_cache_stale(conn: sqlite3.Connection, expiry_hours: int = 18,
                   fetch_type: str = "daily") -> bool:
    """Return True if cache is older than expiry_hours or doesn't exist."""
    last = last_fetch_time(conn, fetch_type)
    if last is None:
        return True
    now = datetime.now(timezone.utc)
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    return (now - last) > timedelta(hours=expiry_hours)
