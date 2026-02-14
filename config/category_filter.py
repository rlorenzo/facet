"""
Facet Category Filter.

Evaluates whether a photo matches a category's filter rules.
"""


# Valid filter fields for v4.0 category-centric config
VALID_NUMERIC_FILTERS = [
    "face_ratio_min", "face_ratio_max",
    "face_count_min", "face_count_max",
    "iso_min", "iso_max",
    "shutter_speed_min", "shutter_speed_max",
    "luminance_min", "luminance_max",
    "focal_length_min", "focal_length_max",
    "f_stop_min", "f_stop_max",
]

VALID_BOOLEAN_FILTERS = [
    "has_face", "is_monochrome", "is_silhouette", "is_group_portrait"
]

VALID_TAG_FILTERS = [
    "required_tags", "excluded_tags", "tag_match_mode"
]

# All valid weight column names (without _percent suffix)
VALID_WEIGHT_COLUMNS = [
    "aesthetic", "face_quality", "eye_sharpness", "tech_sharpness",
    "exposure", "composition", "color", "quality", "contrast",
    "dynamic_range", "isolation", "leading_lines"
]


class CategoryFilter:
    """Evaluates whether a photo matches a category's filter rules.

    Used by v4.0 config schema for config-driven category determination.
    """

    def __init__(self, filter_config: dict):
        """Initialize with filter configuration dict.

        Args:
            filter_config: Dict with filter rules like:
                {
                    "face_ratio_min": 0.05,
                    "has_face": true,
                    "required_tags": ["portrait"],
                    "tag_match_mode": "any"
                }
        """
        self.filters = filter_config or {}

    def matches(self, photo_data: dict) -> bool:
        """Check if photo data matches all filter criteria.

        Args:
            photo_data: Dict with photo metrics and tags. Expected keys:
                - tags: comma-separated string of tags
                - face_count: int
                - face_ratio: float (0-1)
                - is_silhouette: int (0/1)
                - is_group_portrait: int (0/1)
                - is_monochrome: int (0/1)
                - mean_luminance: float (0-1)
                - iso: int or None
                - shutter_speed: float (seconds) or None
                - focal_length: float or None
                - f_stop: float or None

        Returns:
            True if photo matches all filter criteria, False otherwise
        """
        # Empty filters = match everything (fallback category)
        if not self.filters:
            return True

        # Numeric range filters
        # Note: If a filter constraint is defined but the actual value is None,
        # the photo does NOT match (we can't verify the constraint)
        numeric_fields = {
            "face_ratio": photo_data.get("face_ratio"),
            "face_count": photo_data.get("face_count"),
            "iso": photo_data.get("iso"),
            "shutter_speed": photo_data.get("shutter_speed"),
            "luminance": photo_data.get("mean_luminance"),
            "focal_length": photo_data.get("focal_length"),
            "f_stop": photo_data.get("f_stop"),
        }

        for field, actual in numeric_fields.items():
            min_val = self.filters.get(f"{field}_min")
            max_val = self.filters.get(f"{field}_max")

            # If filter is defined but value is None, don't match
            if min_val is not None:
                if actual is None:
                    return False
                if actual < min_val:
                    return False
            if max_val is not None:
                if actual is None:
                    return False
                if actual > max_val:
                    return False

        # Boolean filters
        bool_mappings = {
            "has_face": lambda pd: (pd.get("face_count") or 0) > 0,
            "is_monochrome": lambda pd: bool(pd.get("is_monochrome", 0)),
            "is_silhouette": lambda pd: bool(pd.get("is_silhouette", 0)),
            "is_group_portrait": lambda pd: bool(pd.get("is_group_portrait", 0)),
        }

        for field, getter in bool_mappings.items():
            required = self.filters.get(field)
            if required is not None:
                actual = getter(photo_data)
                if actual != required:
                    return False

        # Tag filters
        required_tags = self.filters.get("required_tags", [])
        excluded_tags = self.filters.get("excluded_tags", [])
        match_mode = self.filters.get("tag_match_mode", "any")

        if required_tags or excluded_tags:
            # Parse photo tags
            tags_str = photo_data.get("tags") or ""
            photo_tags = [t.strip().lower() for t in tags_str.split(",") if t.strip()]

            # Check required tags
            if required_tags:
                required_lower = [t.lower() for t in required_tags]
                if match_mode == "any":
                    if not any(tag in photo_tags for tag in required_lower):
                        return False
                else:  # "all"
                    if not all(tag in photo_tags for tag in required_lower):
                        return False

            # Check excluded tags
            if excluded_tags:
                excluded_lower = [t.lower() for t in excluded_tags]
                if any(tag in photo_tags for tag in excluded_lower):
                    return False

        return True
