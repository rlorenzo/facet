"""
Connection pool for Facet database.

Thread-safe pool of pre-configured SQLite connections.
"""

import sqlite3
import threading
from contextlib import contextmanager
from queue import Queue

from db.connection import DEFAULT_DB_PATH, apply_pragmas


class ConnectionPool:
    """Thread-safe connection pool for SQLite database.

    Maintains a pool of pre-configured database connections to avoid
    repeated PRAGMA setup overhead on each request. Connections are
    configured once when created and reused across requests.

    Usage:
        pool = ConnectionPool('photo_scores_pro.db', size=5)
        conn = pool.get_connection()
        try:
            # Use connection
            cursor = conn.execute("SELECT ...")
        finally:
            pool.return_connection(conn)

        # Or use as context manager:
        with pool.connection() as conn:
            cursor = conn.execute("SELECT ...")
    """

    def __init__(self, db_path=DEFAULT_DB_PATH, size=5, row_factory=True,
                 mmap_size_mb=None, cache_size_mb=None):
        """Initialize connection pool.

        Args:
            db_path: Path to SQLite database
            size: Number of connections to maintain in pool
            row_factory: If True, configure connections with Row factory
            mmap_size_mb: Override mmap_size (MB). None = use config default.
            cache_size_mb: Override cache_size (MB). None = use config default.
        """
        self.db_path = db_path
        self.size = size
        self.row_factory = row_factory
        self._mmap_size_mb = mmap_size_mb
        self._cache_size_mb = cache_size_mb
        self._pool = Queue(maxsize=size)
        self._lock = threading.Lock()
        self._initialized = False

    def _create_connection(self):
        """Create a new pre-configured database connection."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        apply_pragmas(conn, mmap_size_mb=self._mmap_size_mb, cache_size_mb=self._cache_size_mb)
        if self.row_factory:
            conn.row_factory = sqlite3.Row
        return conn

    def _initialize_pool(self):
        """Lazily initialize the connection pool."""
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            for _ in range(self.size):
                self._pool.put(self._create_connection())
            self._initialized = True

    def get_connection(self, timeout=30):
        """Get a connection from the pool.

        Args:
            timeout: Seconds to wait for an available connection

        Returns:
            sqlite3.Connection from the pool

        Raises:
            queue.Empty: If no connection available within timeout
        """
        self._initialize_pool()
        return self._pool.get(timeout=timeout)

    def return_connection(self, conn):
        """Return a connection to the pool.

        Args:
            conn: Connection to return (must be from this pool)
        """
        try:
            # Reset any uncommitted transaction state
            conn.rollback()
            self._pool.put_nowait(conn)
        except Exception:
            # If pool is full or connection is bad, close it
            try:
                conn.close()
            except Exception:
                pass

    @contextmanager
    def connection(self):
        """Context manager for pool connections.

        Usage:
            with pool.connection() as conn:
                conn.execute("SELECT ...")
        """
        conn = self.get_connection()
        try:
            yield conn
        finally:
            self.return_connection(conn)

    def close_all(self):
        """Close all connections in the pool."""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except Exception:
                pass
        self._initialized = False


# Global connection pool (initialized on first use)
_connection_pool = None


def get_pool(db_path=DEFAULT_DB_PATH, size=5, mmap_size_mb=None, cache_size_mb=None):
    """Get or create the global connection pool.

    Args:
        db_path: Database path (only used on first call)
        size: Pool size (only used on first call)
        mmap_size_mb: Override mmap_size (MB, only used on first call)
        cache_size_mb: Override cache_size (MB, only used on first call)

    Returns:
        ConnectionPool instance
    """
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = ConnectionPool(db_path, size=size,
            mmap_size_mb=mmap_size_mb, cache_size_mb=cache_size_mb)
    return _connection_pool


def get_pooled_connection():
    """Get a connection from the global pool.

    Returns:
        Context manager yielding a pooled connection
    """
    return get_pool().connection()
