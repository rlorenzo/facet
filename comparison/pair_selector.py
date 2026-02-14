"""
Pair selection strategies for pairwise photo comparison.
"""

import random
from typing import Optional, Dict

from db import DEFAULT_DB_PATH, get_connection


class PairSelector:
    """Selects pairs of photos for human comparison using various strategies."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path

    def get_next_pair(
        self,
        strategy: str = 'uncertainty',
        category: Optional[str] = None,
        exclude_compared: bool = True
    ) -> Optional[Dict]:
        """
        Select the next pair of photos for comparison.

        Args:
            strategy: Selection strategy - 'uncertainty', 'boundary', 'active', 'random'
            category: Filter photos by category (e.g., 'portrait', 'others')
            exclude_compared: Skip pairs that have already been compared

        Returns:
            Dict with 'a' and 'b' photo paths, or None if no pairs available
        """
        strategies = {
            'uncertainty': self._select_uncertainty,
            'boundary': self._select_boundary,
            'active': self._select_active_learning,
            'random': self._select_random,
        }

        selector = strategies.get(strategy, self._select_uncertainty)
        return selector(category, exclude_compared)

    def _get_compared_pairs(self, conn) -> set:
        """Get set of already-compared pairs (normalized as tuples)."""
        cursor = conn.execute("""
            SELECT photo_a_path, photo_b_path FROM comparisons
        """)
        pairs = set()
        for row in cursor:
            # Normalize to (min, max) to avoid duplicate comparisons in reverse order
            pair = tuple(sorted([row[0], row[1]]))
            pairs.add(pair)
        return pairs

    def _select_uncertainty(
        self,
        category: Optional[str],
        exclude_compared: bool
    ) -> Optional[Dict]:
        """
        Select pairs with similar aggregate scores (high uncertainty).

        Pairs with similar scores are harder to rank, so human input is most valuable.
        """
        with get_connection(self.db_path) as conn:
            compared_pairs = self._get_compared_pairs(conn) if exclude_compared else set()

            # Get photos with aggregate scores, ordered by score
            where_clause = "WHERE aggregate IS NOT NULL"
            params = []
            if category:
                where_clause += " AND category = ?"
                params.append(category)

            cursor = conn.execute(f"""
                SELECT path, aggregate FROM photos
                {where_clause}
                ORDER BY aggregate DESC
            """, params)

            photos = [(row['path'], row['aggregate']) for row in cursor]

            if len(photos) < 2:
                return None

            # Find pairs with smallest score difference
            best_pair = None
            best_diff = float('inf')

            # Sample adjacent pairs in sorted order (most similar scores)
            for i in range(len(photos) - 1):
                path_a, score_a = photos[i]
                path_b, score_b = photos[i + 1]

                # Skip if already compared
                normalized = tuple(sorted([path_a, path_b]))
                if exclude_compared and normalized in compared_pairs:
                    continue

                diff = abs(score_a - score_b)
                if diff < best_diff:
                    best_diff = diff
                    best_pair = {'a': path_a, 'b': path_b, 'score_a': score_a, 'score_b': score_b}

            return best_pair

    def _select_boundary(
        self,
        category: Optional[str],
        exclude_compared: bool
    ) -> Optional[Dict]:
        """
        Select pairs around the score boundary (6-8 range).

        These photos are in the "ambiguous" quality zone where scoring
        precision matters most.
        """
        with get_connection(self.db_path) as conn:
            compared_pairs = self._get_compared_pairs(conn) if exclude_compared else set()

            # Get photos in the 6-8 aggregate range
            where_clause = "WHERE aggregate BETWEEN 5.5 AND 8.5"
            params = []
            if category:
                where_clause += " AND category = ?"
                params.append(category)

            cursor = conn.execute(f"""
                SELECT path, aggregate FROM photos
                {where_clause}
                ORDER BY RANDOM()
                LIMIT 100
            """, params)

            photos = [(row['path'], row['aggregate']) for row in cursor]

            if len(photos) < 2:
                # Fallback to uncertainty selection
                return self._select_uncertainty(category, exclude_compared)

            # Try to find an uncomparied pair
            for _ in range(50):  # Max attempts
                idx_a, idx_b = random.sample(range(len(photos)), 2)
                path_a, score_a = photos[idx_a]
                path_b, score_b = photos[idx_b]

                normalized = tuple(sorted([path_a, path_b]))
                if not exclude_compared or normalized not in compared_pairs:
                    return {'a': path_a, 'b': path_b, 'score_a': score_a, 'score_b': score_b}

            return None

    def _select_active_learning(
        self,
        category: Optional[str],
        exclude_compared: bool
    ) -> Optional[Dict]:
        """
        Select photos with low comparison counts (active learning).

        Prioritizes photos that have been compared fewer times to ensure
        all photos contribute to the learned scores.
        """
        with get_connection(self.db_path) as conn:
            compared_pairs = self._get_compared_pairs(conn) if exclude_compared else set()

            # Get comparison counts per photo
            cursor = conn.execute("""
                SELECT photo_path, comparison_count FROM learned_scores
                ORDER BY comparison_count ASC
            """)
            comparison_counts = {row[0]: row[1] for row in cursor}

            # Get photos, prefer those with low comparison counts
            where_clause = "WHERE aggregate IS NOT NULL"
            params = []
            if category:
                where_clause += " AND category = ?"
                params.append(category)

            cursor = conn.execute(f"""
                SELECT path, aggregate FROM photos
                {where_clause}
            """, params)

            photos = [(row['path'], row['aggregate']) for row in cursor]

            if len(photos) < 2:
                return None

            # Sort by comparison count (ascending)
            photos_with_counts = [
                (path, score, comparison_counts.get(path, 0))
                for path, score in photos
            ]
            photos_with_counts.sort(key=lambda x: x[2])

            # Select from the least-compared photos
            candidates = photos_with_counts[:min(50, len(photos_with_counts))]

            for _ in range(50):
                idx_a, idx_b = random.sample(range(len(candidates)), 2)
                path_a, score_a, _ = candidates[idx_a]
                path_b, score_b, _ = candidates[idx_b]

                normalized = tuple(sorted([path_a, path_b]))
                if not exclude_compared or normalized not in compared_pairs:
                    return {'a': path_a, 'b': path_b, 'score_a': score_a, 'score_b': score_b}

            return None

    def _select_random(
        self,
        category: Optional[str],
        exclude_compared: bool
    ) -> Optional[Dict]:
        """
        Select a random pair of photos.
        """
        with get_connection(self.db_path) as conn:
            compared_pairs = self._get_compared_pairs(conn) if exclude_compared else set()

            where_clause = "WHERE aggregate IS NOT NULL"
            params = []
            if category:
                where_clause += " AND category = ?"
                params.append(category)

            cursor = conn.execute(f"""
                SELECT path, aggregate FROM photos
                {where_clause}
                ORDER BY RANDOM()
                LIMIT 100
            """, params)

            photos = [(row['path'], row['aggregate']) for row in cursor]

            if len(photos) < 2:
                return None

            for _ in range(50):
                idx_a, idx_b = random.sample(range(len(photos)), 2)
                path_a, score_a = photos[idx_a]
                path_b, score_b = photos[idx_b]

                normalized = tuple(sorted([path_a, path_b]))
                if not exclude_compared or normalized not in compared_pairs:
                    return {'a': path_a, 'b': path_b, 'score_a': score_a, 'score_b': score_b}

            return None
