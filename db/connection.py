"""
Database connection utilities for Facet.

Provides connection creation, PRAGMA configuration, and context manager.
"""

import json
import os
import sqlite3
from contextlib import contextmanager

DEFAULT_DB_PATH = os.environ.get('DB_PATH', 'photo_scores_pro.db')
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scoring_config.json')


def get_pragma_values():
    """Read mmap_size and cache_size from scoring_config.json performance section."""
    mmap_size_mb = 256
    cache_size_mb = 64
    try:
        with open(_CONFIG_PATH, 'r') as f:
            config = json.load(f)
        perf = config.get('performance', {})
        mmap_size_mb = perf.get('mmap_size_mb', mmap_size_mb)
        cache_size_mb = perf.get('cache_size_mb', cache_size_mb)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass
    return {
        'mmap_size': mmap_size_mb * 1024 * 1024,
        'cache_size_kb': cache_size_mb * 1000,  # negative KB for PRAGMA cache_size
    }


def apply_pragmas(conn, mmap_size_mb=None, cache_size_mb=None):
    """Apply standard PRAGMA settings to a connection.

    Args:
        conn: SQLite connection
        mmap_size_mb: Override mmap_size (MB). None = use config default.
        cache_size_mb: Override cache_size (MB). None = use config default.
    """
    pv = get_pragma_values()
    mmap_bytes = mmap_size_mb * 1024 * 1024 if mmap_size_mb is not None else pv['mmap_size']
    cache_kb = cache_size_mb * 1000 if cache_size_mb is not None else pv['cache_size_kb']
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute(f"PRAGMA cache_size = -{cache_kb}")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute(f"PRAGMA mmap_size = {mmap_bytes}")


@contextmanager
def get_connection(db_path=DEFAULT_DB_PATH, row_factory=True):
    """
    Context manager for database connections with WAL mode.

    Args:
        db_path: Path to the SQLite database file
        row_factory: If True, set row_factory to sqlite3.Row for dict-like access

    Yields:
        sqlite3.Connection configured with WAL mode and busy timeout
    """
    conn = sqlite3.connect(db_path)
    apply_pragmas(conn)
    if row_factory:
        conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()
