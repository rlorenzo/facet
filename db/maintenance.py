"""
Database maintenance functions for Facet.

VACUUM, ANALYZE, optimization, and viewer database export.
"""

import os
import sqlite3
from io import BytesIO


def vacuum_database(db_path='photo_scores_pro.db', verbose=True):
    """Run VACUUM to reclaim space and defragment the database.

    Args:
        db_path: Path to SQLite database
        verbose: If True, print progress

    Returns:
        Tuple of (old_size, new_size) in bytes
    """
    old_size = os.path.getsize(db_path) if os.path.exists(db_path) else 0

    if verbose:
        print(f"Running VACUUM on {db_path}...")
        print(f"  Before: {old_size / 1024 / 1024:.2f} MB")

    conn = sqlite3.connect(db_path)
    conn.execute("VACUUM")
    conn.close()

    new_size = os.path.getsize(db_path)

    if verbose:
        print(f"  After: {new_size / 1024 / 1024:.2f} MB")
        saved = old_size - new_size
        if saved > 0:
            print(f"  Saved: {saved / 1024 / 1024:.2f} MB ({saved / old_size * 100:.1f}%)")
        else:
            print("  No space reclaimed (database was already compacted)")

    return old_size, new_size


def analyze_database(db_path='photo_scores_pro.db', verbose=True):
    """Run ANALYZE to update query planner statistics.

    Args:
        db_path: Path to SQLite database
        verbose: If True, print progress
    """
    if verbose:
        print(f"Running ANALYZE on {db_path}...")

    conn = sqlite3.connect(db_path)
    conn.execute("ANALYZE")
    conn.close()

    if verbose:
        print("  Query planner statistics updated.")


def optimize_database(db_path='photo_scores_pro.db', verbose=True):
    """Run VACUUM + ANALYZE for full database optimization.

    Args:
        db_path: Path to SQLite database
        verbose: If True, print progress
    """
    if verbose:
        print(f"Optimizing database: {db_path}")
        print()

    vacuum_database(db_path, verbose)
    if verbose:
        print()
    analyze_database(db_path, verbose)

    if verbose:
        print()
        print("Database optimization complete.")


def cleanup_orphaned_persons(db_path='photo_scores_pro.db', verbose=True):
    """Delete persons with no assigned faces.

    Args:
        db_path: Path to SQLite database
        verbose: If True, print progress

    Returns:
        Number of persons deleted
    """
    if verbose:
        print(f"Cleaning up orphaned persons in {db_path}...")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Count orphaned persons before deletion
    cursor.execute("""
        SELECT COUNT(*) FROM persons
        WHERE id NOT IN (SELECT DISTINCT person_id FROM faces WHERE person_id IS NOT NULL)
    """)
    count = cursor.fetchone()[0]

    if count > 0:
        cursor.execute("""
            DELETE FROM persons
            WHERE id NOT IN (SELECT DISTINCT person_id FROM faces WHERE person_id IS NOT NULL)
        """)
        conn.commit()

    conn.close()

    if verbose:
        if count > 0:
            print(f"  Deleted {count} orphaned person(s)")
        else:
            print("  No orphaned persons found")

    return count


def export_viewer_db(source_db='photo_scores_pro.db', output_path=None, thumbnail_size=320, verbose=True):
    """Export a lightweight database for viewer-only deployment.

    Creates a stripped-down copy suitable for low-memory NAS devices by:
    - Removing unused BLOB columns (clip_embedding, histogram_data, face embeddings)
    - Downsizing photo thumbnails from 640px to the specified size
    - Running VACUUM + ANALYZE on the result

    Args:
        source_db: Path to the source database
        output_path: Output path (default: photo_scores_viewer.db)
        thumbnail_size: Max thumbnail dimension in pixels (default: 320)
        verbose: If True, print progress

    Returns:
        Tuple of (source_size, output_size) in bytes
    """
    from PIL import Image

    if output_path is None:
        output_path = 'photo_scores_viewer.db'

    if not os.path.exists(source_db):
        raise FileNotFoundError(f"Source database not found: {source_db}")

    if os.path.abspath(source_db) == os.path.abspath(output_path):
        raise ValueError("Output path cannot be the same as source database")

    source_size = os.path.getsize(source_db)
    if verbose:
        print(f"Exporting viewer database from {source_db}")
        print(f"  Source: {source_size / 1024 / 1024:.1f} MB")

    # Remove existing output file
    if os.path.exists(output_path):
        os.remove(output_path)

    # Use sqlite3.backup() for atomic, WAL-safe copy
    if verbose:
        print("  Copying database...")
    src_conn = sqlite3.connect(source_db)
    dst_conn = sqlite3.connect(output_path)
    src_conn.backup(dst_conn)
    src_conn.close()

    # Strip unused columns from the copy
    if verbose:
        print("  Stripping unused BLOB columns...")

    # photos: clip_embedding, histogram_data, raw_sharpness_variance
    dst_conn.execute("UPDATE photos SET clip_embedding = NULL, histogram_data = NULL, raw_sharpness_variance = NULL")
    dst_conn.commit()

    # faces: embedding (NOT NULL constraint — use empty blob), landmark_2d_106
    dst_conn.execute("UPDATE faces SET embedding = zeroblob(0), landmark_2d_106 = NULL")
    dst_conn.commit()

    # Downsize photo thumbnails
    if verbose:
        row = dst_conn.execute("SELECT COUNT(*) FROM photos WHERE thumbnail IS NOT NULL").fetchone()
        total = row[0]
        print(f"  Downsizing {total} thumbnails to {thumbnail_size}px...")

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    batch_size = 500
    offset = 0
    resized = 0

    while True:
        rows = dst_conn.execute(
            "SELECT path, thumbnail FROM photos WHERE thumbnail IS NOT NULL LIMIT ? OFFSET ?",
            (batch_size, offset)
        ).fetchall()
        if not rows:
            break

        updates = []
        for path, thumb_bytes in rows:
            if thumb_bytes is None:
                continue
            try:
                img = Image.open(BytesIO(thumb_bytes))
                if max(img.size) > thumbnail_size:
                    img.thumbnail((thumbnail_size, thumbnail_size), Image.LANCZOS)
                    buf = BytesIO()
                    img.save(buf, format='JPEG', quality=80)
                    updates.append((buf.getvalue(), path))
                    resized += 1
            except Exception:
                pass  # Skip corrupt thumbnails

        if updates:
            dst_conn.executemany("UPDATE photos SET thumbnail = ? WHERE path = ?", updates)
            dst_conn.commit()

        offset += batch_size
        if verbose and not use_tqdm:
            print(f"    Processed {offset} thumbnails...")

    if verbose:
        print(f"    Resized {resized} thumbnails")

    # Downsize face thumbnails
    if verbose:
        row = dst_conn.execute("SELECT COUNT(*) FROM faces WHERE face_thumbnail IS NOT NULL").fetchone()
        face_total = row[0]
        print(f"  Downsizing {face_total} face thumbnails...")

    offset = 0
    face_resized = 0

    while True:
        rows = dst_conn.execute(
            "SELECT id, face_thumbnail FROM faces WHERE face_thumbnail IS NOT NULL LIMIT ? OFFSET ?",
            (batch_size, offset)
        ).fetchall()
        if not rows:
            break

        updates = []
        for face_id, thumb_bytes in rows:
            if thumb_bytes is None:
                continue
            try:
                img = Image.open(BytesIO(thumb_bytes))
                # Face thumbnails are small; only resize if larger than thumbnail_size
                if max(img.size) > thumbnail_size:
                    img.thumbnail((thumbnail_size, thumbnail_size), Image.LANCZOS)
                    buf = BytesIO()
                    img.save(buf, format='JPEG', quality=80)
                    updates.append((buf.getvalue(), face_id))
                    face_resized += 1
            except Exception:
                pass

        if updates:
            dst_conn.executemany("UPDATE faces SET face_thumbnail = ? WHERE id = ?", updates)
            dst_conn.commit()

        offset += batch_size

    if verbose:
        print(f"    Resized {face_resized} face thumbnails")

    # VACUUM + ANALYZE
    if verbose:
        print("  Running VACUUM...")
    dst_conn.execute("VACUUM")
    if verbose:
        print("  Running ANALYZE...")
    dst_conn.execute("ANALYZE")
    dst_conn.close()

    output_size = os.path.getsize(output_path)
    saved = source_size - output_size
    if verbose:
        print(f"\nResult:")
        print(f"  Source:  {source_size / 1024 / 1024:.1f} MB")
        print(f"  Output:  {output_path} ({output_size / 1024 / 1024:.1f} MB)")
        print(f"  Saved:   {saved / 1024 / 1024:.1f} MB ({saved / source_size * 100:.1f}%)")

    return source_size, output_size
