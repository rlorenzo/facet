"""
Facet utilities package.

Re-exports all public functions and classes for backwards-compatible imports.
"""

from utils.image_loading import load_image_from_path, load_image_for_face_crop, _rawpy_lock
from utils.image_transforms import generate_photo_thumbnail, crop_face_with_padding
from utils.embedding import embedding_to_bytes, bytes_to_embedding
from utils.tags import tags_to_string, string_to_tags, get_tag_params
from utils.detection import (
    detect_silhouette, get_shared_constants,
    DEFAULT_PHOTO_THUMBNAIL_SIZE, DEFAULT_PHOTO_THUMBNAIL_QUALITY,
    DEFAULT_FACE_PADDING_RATIO, DEFAULT_FACE_THUMBNAIL_SIZE,
    DEFAULT_FACE_THUMBNAIL_QUALITY,
)
from utils.burst import IncrementalBurstProcessor
from utils.device import (
    get_best_device, has_gpu, is_cuda, is_mps, safe_empty_cache,
    get_gpu_memory_bytes, get_gpu_memory_gb, get_gpu_allocated_bytes,
    get_gpu_reserved_bytes, get_onnx_providers, get_insightface_ctx_id,
    get_device_map_or_device, is_oom_error,
)
