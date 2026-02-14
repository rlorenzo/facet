"""
Smoke tests for refactored code — ensures all extracted/modified functions
and methods can be called without throwing errors.

Run: python3 -m pytest test_refactoring.py -v
  or: python3 test_refactoring.py
"""

import os
import sys
import sqlite3
import tempfile
import json
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Ensure project root is on sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)


# ============================================================
# Tier 1: database.get_connection() centralization
# ============================================================

class TestDatabaseGetConnection(unittest.TestCase):
    """Test that database.get_connection works as a context manager."""

    def test_get_connection_default(self):
        from db import get_connection
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        try:
            with get_connection(db_path) as conn:
                # row_factory=True by default -> sqlite3.Row
                self.assertEqual(conn.row_factory, sqlite3.Row)
                conn.execute("CREATE TABLE test (id INTEGER)")
                conn.execute("INSERT INTO test VALUES (1)")
                row = conn.execute("SELECT id FROM test").fetchone()
                # sqlite3.Row supports key access
                self.assertEqual(row['id'], 1)
                conn.commit()
        finally:
            os.unlink(db_path)

    def test_get_connection_no_row_factory(self):
        from db import get_connection
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        try:
            with get_connection(db_path, row_factory=False) as conn:
                self.assertIsNone(conn.row_factory)
                conn.execute("CREATE TABLE test (id INTEGER)")
                conn.execute("INSERT INTO test VALUES (1)")
                row = conn.execute("SELECT id FROM test").fetchone()
                # Tuple access
                self.assertEqual(row[0], 1)
                conn.commit()
        finally:
            os.unlink(db_path)

    def test_get_connection_sets_pragmas(self):
        from db import get_connection
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        try:
            with get_connection(db_path, row_factory=False) as conn:
                journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
                self.assertEqual(journal, 'wal')
                busy = conn.execute("PRAGMA busy_timeout").fetchone()[0]
                self.assertEqual(busy, 5000)
                fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
                self.assertEqual(fk, 1)
        finally:
            os.unlink(db_path)


# ============================================================
# Tier 2: Cached config loading in viewer.py
# ============================================================

class TestViewerConfigCaching(unittest.TestCase):
    """Test that viewer config functions don't re-read the file."""

    def test_is_edition_enabled_returns_bool(self):
        # This should work without any file I/O since it reads VIEWER_CONFIG
        from viewer import is_edition_enabled
        result = is_edition_enabled()
        self.assertIsInstance(result, bool)

    def test_get_comparison_mode_settings_returns_dict(self):
        from viewer import get_comparison_mode_settings
        result = get_comparison_mode_settings()
        self.assertIsInstance(result, dict)
        self.assertIn('min_comparisons_for_optimization', result)
        self.assertIn('pair_selection_strategy', result)
        self.assertIn('show_current_scores', result)

    def test_viewer_config_loaded(self):
        from viewer import VIEWER_CONFIG
        self.assertIsInstance(VIEWER_CONFIG, dict)


# ============================================================
# Tier 3: Tag conversion & embedding helpers
# ============================================================

class TestTagHelpers(unittest.TestCase):
    """Test that image_utils tag/embedding helpers work correctly."""

    def test_tags_to_string(self):
        from utils import tags_to_string
        self.assertIsNone(tags_to_string([]))
        self.assertIsNone(tags_to_string(None))
        self.assertEqual(tags_to_string(['a', 'b']), 'a,b')

    def test_string_to_tags(self):
        from utils import string_to_tags
        self.assertEqual(string_to_tags(''), [])
        self.assertEqual(string_to_tags(None), [])
        self.assertEqual(string_to_tags('a, b, c'), ['a', 'b', 'c'])

    def test_embedding_to_bytes_and_back(self):
        import numpy as np
        from utils import embedding_to_bytes, bytes_to_embedding
        arr = np.random.randn(512).astype(np.float32)
        data = embedding_to_bytes(arr)
        self.assertIsInstance(data, bytes)
        result = bytes_to_embedding(data, dim=512)
        self.assertIsNotNone(result)
        np.testing.assert_array_almost_equal(arr, result)

    def test_bytes_to_embedding_none(self):
        from utils import bytes_to_embedding
        self.assertIsNone(bytes_to_embedding(None))

    def test_bytes_to_embedding_wrong_dim(self):
        import numpy as np
        from utils import bytes_to_embedding
        data = np.zeros(512, dtype=np.float32).tobytes()
        self.assertIsNone(bytes_to_embedding(data, dim=768))

    def test_tagger_no_duplicate_functions(self):
        """Ensure tags_to_string/string_to_tags are NOT in tagger.py anymore."""
        from models import tagger
        self.assertFalse(hasattr(tagger, 'tags_to_string'))
        self.assertFalse(hasattr(tagger, 'string_to_tags'))


# ============================================================
# Tier 4: load_image_for_face_crop
# ============================================================

class TestLoadImageForFaceCrop(unittest.TestCase):
    """Test that load_image_for_face_crop handles missing files gracefully."""

    def test_nonexistent_file(self):
        from utils import load_image_for_face_crop
        img, sx, sy = load_image_for_face_crop('/nonexistent/path.jpg')
        self.assertIsNone(img)
        self.assertEqual(sx, 1.0)
        self.assertEqual(sy, 1.0)

    def test_nonexistent_raw(self):
        from utils import load_image_for_face_crop
        img, sx, sy = load_image_for_face_crop('/nonexistent/path.cr2')
        self.assertIsNone(img)
        self.assertEqual(sx, 1.0)
        self.assertEqual(sy, 1.0)


# ============================================================
# Tier 5: Viewer helper functions
# ============================================================

class TestViewerTagFilter(unittest.TestCase):
    """Test _add_tag_filter helper."""

    def test_add_tag_filter_no_args(self):
        from viewer import _add_tag_filter
        where = []
        params = []
        _add_tag_filter(where, params)
        self.assertEqual(where, [])
        self.assertEqual(params, [])

    def test_add_tag_filter_single_tag(self):
        from viewer import _add_tag_filter
        where = []
        params = []
        _add_tag_filter(where, params, tag='landscape')
        self.assertEqual(len(where), 1)
        # params contains either 'landscape' (photo_tags) or '%landscape%' (LIKE)
        self.assertTrue(any('landscape' in str(p) for p in params))

    def test_add_tag_filter_require_tags(self):
        from viewer import _add_tag_filter
        where = []
        params = []
        _add_tag_filter(where, params, require_tags='landscape,mountain')
        self.assertEqual(len(where), 1)
        self.assertTrue(len(params) >= 2)

    def test_add_tag_filter_exclude_tags(self):
        from viewer import _add_tag_filter
        where = []
        params = []
        _add_tag_filter(where, params, exclude_tags='art,painting')
        self.assertEqual(len(where), 2)  # One clause per excluded tag
        self.assertEqual(len(params), 2)

    def test_add_tag_filter_exclude_art(self):
        from viewer import _add_tag_filter
        where = []
        params = []
        _add_tag_filter(where, params, exclude_art_tags=['painting', 'sculpture'])
        self.assertEqual(len(where), 1)
        self.assertTrue(len(params) >= 2)


# ============================================================
# Tier 6: Extracted long-function helpers
# ============================================================

class TestSafeFloat(unittest.TestCase):
    """Test _safe_float from photos.py."""

    def test_none(self):
        from processing.scorer import _safe_float
        self.assertEqual(_safe_float(None), 5.0)

    def test_bytes(self):
        from processing.scorer import _safe_float
        self.assertEqual(_safe_float(b'\x00\x01'), 5.0)

    def test_normal_float(self):
        from processing.scorer import _safe_float
        self.assertEqual(_safe_float(7.5), 7.5)

    def test_int(self):
        from processing.scorer import _safe_float
        self.assertEqual(_safe_float(3), 3.0)

    def test_string_float(self):
        from processing.scorer import _safe_float
        self.assertEqual(_safe_float('4.2'), 4.2)

    def test_invalid_string(self):
        from processing.scorer import _safe_float
        self.assertEqual(_safe_float('abc', default=0.0), 0.0)

    def test_out_of_range(self):
        from processing.scorer import _safe_float
        self.assertEqual(_safe_float(999.0), 5.0)

    def test_custom_default(self):
        from processing.scorer import _safe_float
        self.assertEqual(_safe_float(None, default=0.0), 0.0)


class TestCalculateScoringPenalties(unittest.TestCase):
    """Test _calculate_scoring_penalties from photos.py."""

    def test_no_penalties(self):
        from processing.scorer import _calculate_scoring_penalties
        metrics = {
            'noise_sigma': 0,
            'histogram_bimodality': 0,
            'mean_saturation': 0.5,
            'leading_lines_score': 0,
        }
        config = MagicMock()
        config.get_penalty_settings.return_value = {}
        result = _calculate_scoring_penalties(metrics, config)
        self.assertIsInstance(result, dict)
        self.assertEqual(result['noise_penalty'], 0)
        self.assertEqual(result['bimodality_penalty'], 0)
        self.assertEqual(result['oversaturation_penalty'], 0)

    def test_noise_penalty(self):
        from processing.scorer import _calculate_scoring_penalties
        metrics = {
            'noise_sigma': 10.0,
            'histogram_bimodality': 0,
            'mean_saturation': 0.5,
            'leading_lines_score': 0,
        }
        config = MagicMock()
        config.get_penalty_settings.return_value = {}
        result = _calculate_scoring_penalties(metrics, config)
        self.assertGreater(result['noise_penalty'], 0)

    def test_none_config(self):
        from processing.scorer import _calculate_scoring_penalties
        metrics = {'noise_sigma': 0, 'histogram_bimodality': 0, 'mean_saturation': 0, 'leading_lines_score': 0}
        result = _calculate_scoring_penalties(metrics, None)
        self.assertIsInstance(result, dict)
        self.assertIn('noise_penalty', result)

    def test_all_keys_present(self):
        from processing.scorer import _calculate_scoring_penalties
        metrics = {'noise_sigma': 0, 'histogram_bimodality': 0, 'mean_saturation': 0, 'leading_lines_score': 5}
        config = MagicMock()
        config.get_penalty_settings.return_value = {}
        result = _calculate_scoring_penalties(metrics, config)
        for key in ['noise_penalty', 'noise_sigma', 'bimodality_penalty', 'oversaturation_penalty', 'leading_lines', 'leading_lines_blend']:
            self.assertIn(key, result)


class TestCalcStats(unittest.TestCase):
    """Test _calc_stats from config.py."""

    def test_empty(self):
        from config import _calc_stats
        self.assertIsNone(_calc_stats([]))

    def test_single_value(self):
        from config import _calc_stats
        result = _calc_stats([5.0])
        self.assertIsNotNone(result)
        self.assertEqual(result['count'], 1)
        self.assertEqual(result['min'], 5.0)
        self.assertEqual(result['max'], 5.0)
        self.assertEqual(result['avg'], 5.0)

    def test_multiple_values(self):
        from config import _calc_stats
        result = _calc_stats([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        self.assertEqual(result['count'], 11)
        self.assertEqual(result['min'], 1)
        self.assertEqual(result['max'], 11)
        self.assertAlmostEqual(result['avg'], 6.0)
        self.assertIn('p10', result)
        self.assertIn('p50', result)
        self.assertIn('p90', result)

    def test_all_keys_present(self):
        from config import _calc_stats
        result = _calc_stats([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
        for key in ['count', 'min', 'max', 'avg', 'std', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95']:
            self.assertIn(key, result)


# ============================================================
# Tier 7: No unused imports — syntax check
# ============================================================

class TestSyntaxAndImports(unittest.TestCase):
    """Verify all key files parse without SyntaxError."""

    def test_all_files_parse(self):
        import ast
        files = [
            'photos.py', 'face_clustering.py', 'config.py', 'viewer.py',
            'database.py', 'tagger.py',
        ]
        for fname in files:
            fpath = os.path.join(_script_dir, fname)
            if os.path.exists(fpath):
                with open(fpath) as f:
                    source = f.read()
                try:
                    ast.parse(source)
                except SyntaxError as e:
                    self.fail(f"SyntaxError in {fname}: {e}")


# ============================================================
# Tier 8: Config magic numbers
# ============================================================

class TestScoringConfigLimits(unittest.TestCase):
    """Test get_scoring_limits and config magic numbers."""

    def test_get_scoring_limits(self):
        from config import ScoringConfig
        cfg = ScoringConfig()
        limits = cfg.get_scoring_limits()
        self.assertIsInstance(limits, dict)
        self.assertIn('score_min', limits)
        self.assertIn('score_max', limits)
        self.assertIn('score_precision', limits)
        self.assertEqual(limits['score_min'], 0.0)
        self.assertEqual(limits['score_max'], 10.0)
        self.assertEqual(limits['score_precision'], 2)

    def test_face_clustering_config_has_merge_threshold(self):
        from config import ScoringConfig
        cfg = ScoringConfig()
        face_settings = cfg.get_face_clustering_settings()
        self.assertIn('merge_threshold', face_settings)
        self.assertAlmostEqual(face_settings['merge_threshold'], 0.6)

    def test_face_processing_config_has_crop_padding(self):
        from config import ScoringConfig
        cfg = ScoringConfig()
        face_settings = cfg.get_face_processing_settings()
        self.assertIn('crop_padding', face_settings)
        self.assertAlmostEqual(face_settings['crop_padding'], 0.3)


# ============================================================
# Cross-cutting: database module functions
# ============================================================

class TestDatabaseModuleFunctions(unittest.TestCase):
    """Test database.py functions use get_connection properly."""

    def test_init_database(self):
        from db import init_database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        try:
            init_database(db_path)
            # Verify tables exist
            conn = sqlite3.connect(db_path)
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            conn.close()
            self.assertIn('photos', tables)
            self.assertIn('faces', tables)
            self.assertIn('persons', tables)
        finally:
            os.unlink(db_path)

    def test_get_schema_info(self):
        from db import get_schema_info
        info = get_schema_info()
        self.assertIn('photos_columns', info)
        self.assertIn('indexes', info)
        self.assertIn('column_names', info)

    def test_get_photo_tags_count(self):
        from db import get_photo_tags_count, init_database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        try:
            init_database(db_path)
            count = get_photo_tags_count(db_path)
            self.assertEqual(count, 0)
        finally:
            os.unlink(db_path)

    def test_refresh_stats_cache(self):
        from db import refresh_stats_cache, init_database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        try:
            init_database(db_path)
            stats = refresh_stats_cache(db_path, verbose=False)
            self.assertIsInstance(stats, dict)
            self.assertIn('total_photos', stats)
            self.assertEqual(stats['total_photos'], 0)
        finally:
            os.unlink(db_path)

    def test_get_cached_stat(self):
        from db import get_cached_stat, init_database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        try:
            init_database(db_path)
            value, is_fresh = get_cached_stat(db_path, key='total_photos')
            # No cache populated yet
            self.assertIsNone(value)
            self.assertFalse(is_fresh)
        finally:
            os.unlink(db_path)

    def test_get_stats_cache_info(self):
        from db import get_stats_cache_info, init_database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        try:
            init_database(db_path)
            info = get_stats_cache_info(db_path)
            self.assertIsInstance(info, dict)
        finally:
            os.unlink(db_path)


# ============================================================
# IncrementalBurstProcessor.finalize refactoring
# ============================================================

class TestBurstProcessorFinalize(unittest.TestCase):
    """Test that IncrementalBurstProcessor.finalize works with both paths."""

    def test_finalize_with_conn(self):
        from utils import IncrementalBurstProcessor
        from db import init_database
        from config import ScoringConfig

        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        try:
            init_database(db_path)
            cfg = ScoringConfig()
            proc = IncrementalBurstProcessor(db_path, cfg)
            # Finalize with explicit connection
            conn = sqlite3.connect(db_path)
            result = proc.finalize(conn=conn)
            self.assertEqual(result, 0)  # No bursts
            conn.close()
        finally:
            os.unlink(db_path)

    def test_finalize_without_conn(self):
        from utils import IncrementalBurstProcessor
        from db import init_database
        from config import ScoringConfig

        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        try:
            init_database(db_path)
            cfg = ScoringConfig()
            proc = IncrementalBurstProcessor(db_path, cfg)
            # Finalize without connection (uses get_connection internally)
            result = proc.finalize()
            self.assertEqual(result, 0)
        finally:
            os.unlink(db_path)


if __name__ == '__main__':
    unittest.main()
