"""Session-state helpers for the Streamlit UI."""

from typing import Any, Dict

import streamlit as st


DEFAULT_STATE: Dict[str, Any] = {
    "user_id": "demo_user",
    "search_results": [],
    "selected_product": None,
    "recommendations": [],
    "comparison": {"generic": [], "personalized": []},
    "interaction_count": 0,
    "last_interaction": None,
    "user_state": None,
    "interaction_history": [],
    "search_feedback": "",
    "client_signature": "",
    "api_client": None,
    "has_run_initial_search": False,
}


RUNTIME_KEYS = [
    "search_results",
    "selected_product",
    "recommendations",
    "comparison",
    "interaction_count",
    "last_interaction",
    "user_state",
    "interaction_history",
    "search_feedback",
    "api_client",
    "client_signature",
    "has_run_initial_search",
]


def init_session_state() -> None:
    """Initialize required session keys exactly once."""
    for key, value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_runtime_state() -> None:
    """Clear only run-time values while preserving UI preferences."""
    for key in RUNTIME_KEYS:
        st.session_state[key] = DEFAULT_STATE[key]
