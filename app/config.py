"""Configuration values used by the Streamlit frontend."""

from dataclasses import dataclass
from typing import Dict

DEFAULT_API_BASE_URL = "http://localhost:8000"
DEFAULT_USER_ID = "demo_user"

SEARCH_MODES = ("hybrid", "semantic", "keyword")

COLOR_OPTIONS = [
    "All",
    "Black",
    "White",
    "Blue",
    "Red",
    "Green",
    "Pink",
    "Grey",
    "Beige",
    "Brown",
]

DEPARTMENT_OPTIONS = [
    "All",
    "Ladieswear",
    "Menswear",
    "Divided",
    "Kids",
    "Sports",
]

@dataclass
class SidebarSettings:
    """All controls selected from the sidebar."""

    user_id: str
    use_stub: bool
    api_base_url: str
    search_mode: str
    alpha: float
    filters: Dict[str, str]
    search_limit: int
    recommend_k: int
    show_before_after: bool
    reset_clicked: bool
