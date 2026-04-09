"""Configuration values used by the Streamlit frontend."""

import os
from dataclasses import dataclass
from typing import Dict

# Modal deployed URL should be provided via environment variable; 
# fallback to a placeholder or localhost for testing.
DEFAULT_API_BASE_URL = os.environ.get("MODAL_API_BASE_URL", "https://huytq2810--fashion-recsys.modal.run")
USE_STUB_API = os.environ.get("USE_STUB_API", "false").lower() == "true"
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
