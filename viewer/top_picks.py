from viewer.config import VIEWER_CONFIG


# --- TOP PICKS SCORE COMPUTATION ---
def get_top_picks_score_sql():
    """Build SQL expression for top_picks_score based on config weights.

    When face_ratio >= min_face_ratio: uses configured weights including face_quality
    Otherwise: redistributes face_quality weight equally to aesthetic and composition
    """
    pt = VIEWER_CONFIG['photo_types']
    weights = pt.get('top_picks_weights', {
        'aggregate_percent': 20,
        'aesthetic_percent': 32,
        'composition_percent': 24,
        'face_quality_percent': 24
    })

    # Minimum face ratio to use face_quality weight (default 20%)
    min_face_ratio = pt.get('top_picks_min_face_ratio', 0.20)

    # Convert percentages to decimals
    agg_w = weights.get('aggregate_percent', 20) / 100.0
    aesthetic_w = weights.get('aesthetic_percent', 32) / 100.0
    comp_w = weights.get('composition_percent', 24) / 100.0
    face_w = weights.get('face_quality_percent', 24) / 100.0

    # For photos without significant faces, redistribute face_quality weight equally
    no_face_aesthetic_w = aesthetic_w + (face_w / 2.0)
    no_face_comp_w = comp_w + (face_w / 2.0)

    return f"""CASE
        WHEN COALESCE(face_ratio, 0) >= {min_face_ratio} THEN
            (COALESCE(aggregate, 0) * {agg_w:.2f} + COALESCE(aesthetic, 0) * {aesthetic_w:.2f} + COALESCE(comp_score, 0) * {comp_w:.2f} + COALESCE(face_quality, 0) * {face_w:.2f})
        ELSE
            (COALESCE(aggregate, 0) * {agg_w:.2f} + COALESCE(aesthetic, 0) * {no_face_aesthetic_w:.2f} + COALESCE(comp_score, 0) * {no_face_comp_w:.2f})
    END"""


def get_top_picks_threshold():
    """Get the minimum score threshold for top picks."""
    return VIEWER_CONFIG['photo_types'].get('top_picks_min_score', 7)
