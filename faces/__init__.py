"""
Facet faces package.

Face clustering, extraction, and thumbnail processing.
"""

from faces.resource_monitor import FaceResourceMonitor
from faces.processor import FaceProcessor
from faces.clusterer import (
    FaceClusterer,
    _check_cuml_available,
    extract_faces_from_existing,
    refill_face_thumbnails,
    run_face_clustering,
)
from faces.merge_analyzer import suggest_person_merges, get_merge_groups
