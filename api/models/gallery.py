"""Pydantic models for gallery endpoints."""

from pydantic import BaseModel
from typing import Optional


class PhotoPerson(BaseModel):
    id: int
    name: str


class Photo(BaseModel):
    path: str
    filename: Optional[str] = None
    date_taken: Optional[str] = None
    date_formatted: Optional[str] = None
    camera_model: Optional[str] = None
    lens_model: Optional[str] = None
    iso: Optional[int] = None
    f_stop: Optional[float] = None
    shutter_speed: Optional[float] = None
    focal_length: Optional[float] = None
    aesthetic: Optional[float] = None
    face_count: Optional[int] = None
    face_quality: Optional[float] = None
    eye_sharpness: Optional[float] = None
    face_sharpness: Optional[float] = None
    face_ratio: Optional[float] = None
    tech_sharpness: Optional[float] = None
    color_score: Optional[float] = None
    exposure_score: Optional[float] = None
    comp_score: Optional[float] = None
    isolation_bonus: Optional[float] = None
    aggregate: Optional[float] = None
    category: Optional[str] = None
    tags: Optional[str] = None
    tags_list: list[str] = []
    composition_pattern: Optional[str] = None
    is_blink: Optional[int] = None
    is_burst_lead: Optional[int] = None
    is_monochrome: Optional[int] = None
    noise_sigma: Optional[float] = None
    contrast_score: Optional[float] = None
    dynamic_range_stops: Optional[float] = None
    mean_saturation: Optional[float] = None
    mean_luminance: Optional[float] = None
    histogram_spread: Optional[float] = None
    power_point_score: Optional[float] = None
    leading_lines_score: Optional[float] = None
    quality_score: Optional[float] = None
    star_rating: Optional[int] = None
    is_favorite: Optional[int] = None
    is_rejected: Optional[int] = None
    duplicate_group_id: Optional[int] = None
    is_duplicate_lead: Optional[int] = None
    top_picks_score: Optional[float] = None
    persons: list[PhotoPerson] = []
    unassigned_faces: int = 0

    model_config = {'from_attributes': True}


class GalleryResponse(BaseModel):
    photos: list[dict]  # Using dict for flexibility with optional columns
    page: int
    total_pages: int
    total_count: int
    has_more: bool
    sort_col: str


class TypeCountItem(BaseModel):
    id: str
    label: str
    count: int


class TypeCountsResponse(BaseModel):
    types: list[TypeCountItem]


class SimilarPhoto(BaseModel):
    path: str
    filename: Optional[str] = None
    similarity: float
    breakdown: dict
    aggregate: Optional[float] = None
    aesthetic: Optional[float] = None
    date_taken: Optional[str] = None


class SimilarPhotosResponse(BaseModel):
    source: str
    weights: dict
    similar: list[SimilarPhoto]
