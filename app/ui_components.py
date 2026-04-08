"""Reusable Streamlit UI components for the recommendation demo."""

from __future__ import annotations

from html import escape
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from app.config import (
    COLOR_OPTIONS,
    DEFAULT_API_BASE_URL,
    DEFAULT_USER_ID,
    DEPARTMENT_OPTIONS,
    SEARCH_MODES,
    SidebarSettings,
)


def render_hero() -> None:
    """Top section with project context and visual identity."""
    st.markdown(
        """
        <div class="fr-hero">
          <h2 style="margin-bottom:0.2rem;">Fashion Discovery + Personalized Re-ranking</h2>
          <p class="fr-caption" style="margin-bottom:0.4rem;">
            Search products with text or image, then personalize recommendations from user interactions.
          </p>
          <span class="fr-tag">Hybrid Search</span>
          <span class="fr-tag">Real-time Personalization</span>
          <span class="fr-tag">API-ready Contracts</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> SidebarSettings:
    """Render all sidebar controls and return values as a typed object."""
    st.sidebar.header("Runtime Settings")

    user_id = st.sidebar.text_input("User ID", value=st.session_state.get("user_id", DEFAULT_USER_ID))
    use_stub = st.sidebar.toggle("Use API stub", value=True)

    api_base_url = st.sidebar.text_input(
        "Backend base URL",
        value=DEFAULT_API_BASE_URL,
        disabled=use_stub,
        help="Used only when stub mode is off.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Search")

    mode_label_map = {"hybrid": "Hybrid", "semantic": "Semantic", "keyword": "Keyword"}
    selected_label = st.sidebar.selectbox(
        "Search mode",
        options=[mode_label_map[m] for m in SEARCH_MODES],
        index=0,
    )
    search_mode = selected_label.lower()

    alpha = st.sidebar.slider(
        "Hybrid alpha",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        disabled=(search_mode != "hybrid"),
    )

    search_limit = st.sidebar.slider("Search results", min_value=8, max_value=40, value=16, step=4)
    recommend_k = st.sidebar.slider("Recommendations", min_value=5, max_value=20, value=10, step=1)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Metadata filters")

    color = st.sidebar.selectbox("Color", options=COLOR_OPTIONS, index=0)
    department = st.sidebar.selectbox("Department", options=DEPARTMENT_OPTIONS, index=0)
    product_type = st.sidebar.text_input("Product type", placeholder="Example: Dress, Jacket")

    filters: Dict[str, str] = {}
    if color != "All":
        filters["colour_group_name"] = color
    if department != "All":
        filters["department_name"] = department
    if product_type.strip():
        filters["product_type_name"] = product_type.strip()

    st.sidebar.markdown("---")
    show_before_after = st.sidebar.checkbox("Show before/after personalization", value=True)
    reset_clicked = st.sidebar.button("Reset local UI state", use_container_width=True)

    return SidebarSettings(
        user_id=user_id.strip() or DEFAULT_USER_ID,
        use_stub=use_stub,
        api_base_url=api_base_url.strip() or DEFAULT_API_BASE_URL,
        search_mode=search_mode,
        alpha=alpha,
        filters=filters,
        search_limit=search_limit,
        recommend_k=recommend_k,
        show_before_after=show_before_after,
        reset_clicked=reset_clicked,
    )


def render_search_form() -> Tuple[str, Optional[bytes], bool]:
    """Render top search form and return query/image/submitted."""
    with st.form("search_form", clear_on_submit=False):
        col_query, col_upload = st.columns([2.6, 1.4])

        with col_query:
            query = st.text_input(
                "Search query",
                placeholder="Try: black blazer, summer dress, street hoodie",
            )

        with col_upload:
            upload = st.file_uploader(
                "Image (optional)",
                type=["jpg", "jpeg", "png"],
            )

        submitted = st.form_submit_button("Search products", use_container_width=True)

    image_bytes = upload.getvalue() if upload is not None else None
    return query.strip(), image_bytes, submitted


def render_feedback(message: str) -> None:
    """Render compact status text for the last API call."""
    if not message:
        return
    st.caption(message)


def render_product_grid(results: List[Dict], columns: int = 4) -> Optional[Dict]:
    """Display result cards and return the selected product when clicked."""
    if not results:
        st.info("No products found. Try another query or relax filters.")
        return None

    selected_product = None
    cols = st.columns(columns)

    for idx, item in enumerate(results):
        col = cols[idx % columns]
        with col:
            _render_product_card(item)
            btn_key = f"select_{item.get('article_id', idx)}"
            if st.button("View details", key=btn_key, use_container_width=True):
                selected_product = item

    return selected_product


def _render_product_card(item: Dict) -> None:
    title = escape(item.get("product_name", "Unknown item"))
    article_id = escape(str(item.get("article_id", "N/A")))
    product_type = escape(item.get("product_type_name", "Unknown"))
    color = escape(item.get("colour_group_name", "Unknown"))
    dept = escape(item.get("department_name", "Unknown"))
    score = float(item.get("score", 0.0))
    price = item.get("price")
    image_path = item.get("image_path") or item.get("image_url")

    price_line = ""
    if price is not None:
        price_line = f"<div class='fr-meta'>Price: ${float(price):.2f}</div>"

    st.markdown(
        f"""
        <div class="fr-card">
          <div class="fr-card-title">{title}</div>
          <div class="fr-meta">ID: {article_id}</div>
          <div class="fr-meta">{product_type} | {color}</div>
          <div class="fr-meta">{dept}</div>
          {price_line}
          <div class="fr-score">Score: {score:.3f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if image_path:
        st.image(image_path, use_container_width=True)


def render_selected_product(product: Dict) -> None:
    """Highlight currently selected product."""
    st.markdown(
        f"""
        <div class="fr-selected">
          <div class="fr-card-title">Selected: {escape(product.get('product_name', 'Unknown item'))}</div>
          <div class="fr-meta">Article: {escape(str(product.get('article_id', 'N/A')))}</div>
          <div class="fr-meta">{escape(product.get('product_type_name', 'Unknown'))} | {escape(product.get('colour_group_name', 'Unknown'))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recommendations(title: str, recommendations: List[Dict], columns: int = 4) -> Optional[Dict]:
    """Render recommendation cards and return clicked recommendation if any."""
    st.subheader(title)
    if not recommendations:
        st.info("No recommendations yet.")
        return None

    selected = None
    cols = st.columns(columns)

    for idx, rec in enumerate(recommendations):
        col = cols[idx % columns]
        with col:
            _render_recommendation_card(rec)
            key = f"rec_select_{rec.get('article_id', idx)}"
            if st.button("View details", key=key, use_container_width=True):
                selected = rec

    return selected


def _render_recommendation_card(rec: Dict) -> None:
    _render_product_card(rec)


def render_before_after(generic: List[Dict], personalized: List[Dict]) -> None:
    """Show side-by-side recommendation ranking for personalization explainability."""
    st.subheader("Before vs After personalization")

    max_len = max(len(generic), len(personalized), 1)
    rows = []
    for i in range(max_len):
        generic_name = generic[i].get("product_name", "") if i < len(generic) else ""
        generic_score = generic[i].get("score", "") if i < len(generic) else ""
        personal_name = personalized[i].get("product_name", "") if i < len(personalized) else ""
        personal_score = personalized[i].get("score", "") if i < len(personalized) else ""
        rows.append(
            {
                "rank": i + 1,
                "generic": generic_name,
                "g_score": generic_score,
                "personalized": personal_name,
                "p_score": personal_score,
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_personalization_panel(user_state: Optional[Dict], last_interaction: Optional[Dict]) -> None:
    """Display counters, adaptation status, and an EMA trend preview."""
    st.subheader("Personalization panel")

    if not user_state or not user_state.get("exists", False):
        st.info("No user profile yet. Interact with products to build personalization.")
        return

    metric_cols = st.columns(4)
    metric_cols[0].metric("Interactions", int(user_state.get("interaction_count", 0)))
    metric_cols[1].metric("Cache", user_state.get("cache_location", "none"))
    metric_cols[2].metric("TTL(s)", user_state.get("redis_ttl_remaining_seconds") or "-")

    adapted_flag = False
    if last_interaction:
        adapted_flag = bool(last_interaction.get("adapted", False))
    metric_cols[3].metric("Last adaptation", "yes" if adapted_flag else "no")

    if last_interaction and adapted_flag:
        latency = last_interaction.get("adaptation_time_ms")
        st.success(f"Triplet adaptation triggered. Latency: {latency} ms")

    ema_vector = user_state.get("ema_vector")
    if ema_vector:
        preview = pd.DataFrame(
            {
                "dim": list(range(min(len(ema_vector), 32))),
                "value": ema_vector[:32],
            }
        )
        st.caption("EMA vector preview (first 32 dimensions)")
        st.line_chart(preview.set_index("dim"), height=180)
