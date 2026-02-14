"""
Detection utilities for Facet.

Silhouette detection and shared constants.
"""


def detect_silhouette(histogram_data, tags, face_count):
    """
    Determine if image is a silhouette based on histogram and CLIP tags.

    Silhouette detection requires:
    - (histogram silhouette OR CLIP silhouette tag) AND human presence

    Args:
        histogram_data: Dict with 'is_silhouette' key from histogram analysis
        tags: Comma-separated tag string or None
        face_count: Number of detected faces

    Returns:
        int: 1 if silhouette detected, 0 otherwise
    """
    histogram_silhouette = histogram_data.get('is_silhouette', 0)
    clip_silhouette = 'silhouette' in tags if tags else False
    has_human = (
        face_count > 0 or
        (any(t in tags for t in ['portrait', 'group']) if tags else False)
    )
    return 1 if ((histogram_silhouette or clip_silhouette) and has_human) else 0


# Shared constants from config (with defaults)
DEFAULT_PHOTO_THUMBNAIL_SIZE = 640
DEFAULT_PHOTO_THUMBNAIL_QUALITY = 80
DEFAULT_FACE_PADDING_RATIO = 0.3
DEFAULT_FACE_THUMBNAIL_SIZE = 128
DEFAULT_FACE_THUMBNAIL_QUALITY = 85


def get_shared_constants(config=None):
    """
    Get shared constants from config or use defaults.

    Args:
        config: Optional ScoringConfig instance

    Returns:
        dict: Shared constants for thumbnails and face cropping
    """
    constants = {
        'photo_thumbnail_size': DEFAULT_PHOTO_THUMBNAIL_SIZE,
        'photo_thumbnail_quality': DEFAULT_PHOTO_THUMBNAIL_QUALITY,
        'face_padding_ratio': DEFAULT_FACE_PADDING_RATIO,
        'face_thumbnail_size': DEFAULT_FACE_THUMBNAIL_SIZE,
        'face_thumbnail_quality': DEFAULT_FACE_THUMBNAIL_QUALITY,
    }

    if config is not None:
        proc_settings = config.get_processing_settings()
        thumbnails = proc_settings.get('thumbnails', {})
        constants.update({
            'photo_thumbnail_size': thumbnails.get('photo_size', constants['photo_thumbnail_size']),
            'photo_thumbnail_quality': thumbnails.get('photo_quality', constants['photo_thumbnail_quality']),
            'face_padding_ratio': thumbnails.get('face_padding_ratio', constants['face_padding_ratio']),
        })

        # Face processing settings
        face_proc = config.get_face_processing_settings()
        constants.update({
            'face_thumbnail_size': face_proc.get('face_thumbnail_size', constants['face_thumbnail_size']),
            'face_thumbnail_quality': face_proc.get('face_thumbnail_quality', constants['face_thumbnail_quality']),
        })

    return constants
