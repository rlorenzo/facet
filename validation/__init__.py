"""
Facet validation package.

Database consistency validation and auto-fix tools.
"""

from validation.validation_result import ValidationResult
from validation.database_validator import (
    DatabaseValidator,
    SCORE_COLUMNS,
    RAW_METRIC_COLUMNS,
    FACE_DEPENDENT_COLUMNS,
    BOOLEAN_COLUMNS,
    VALID_COMPOSITION_PATTERNS,
)
