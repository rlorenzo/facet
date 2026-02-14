"""
Validation result container for database checks.
"""

from typing import Any, Callable, Dict, List, Optional


class ValidationResult:
    """Container for validation check results."""

    def __init__(self, check_name: str, description: str, informational: bool = False):
        self.check_name = check_name
        self.description = description
        self.issues: List[Dict[str, Any]] = []
        self.fixable = False
        self.fix_query: Optional[str] = None
        self.fix_function: Optional[Callable] = None
        self.informational = informational  # True if this is just FYI, not an error

    def add_issue(self, record: Dict[str, Any], details: str = ""):
        self.issues.append({'record': record, 'details': details})

    @property
    def has_issues(self) -> bool:
        return len(self.issues) > 0

    @property
    def count(self) -> int:
        return len(self.issues)
