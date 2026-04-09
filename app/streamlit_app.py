"""Modular Streamlit frontend for fashion search and personalized recommendations."""

from __future__ import annotations

import base64
import csv
import hashlib
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

# Ensure project root is importable when Streamlit is launched from outside repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api_client import APIClientError, build_api_client
from app.config import SidebarSettings
from app.state import init_session_state, reset_runtime_state
from app.style import inject_global_styles
from app.ui_components import (
    render_before_after,
    render_feedback,
    render_hero,
    render_personalization_panel,
    render_product_grid,
    render_recommendations,
    render_search_form,
    render_selected_product,
    render_sidebar,
)


st.set_page_config(
    page_title="Fashion Recommender",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _seed_from_text(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _derive_price(article_id: str) -> float:
    rng = random.Random(_seed_from_text(f"price:{article_id}"))
    return round(12.0 + rng.random() * 138.0, 2)


def _discover_fallback_image_path(images_root: Path) -> str:
    if not images_root.exists():
        return ""

    for shard in sorted(images_root.iterdir()):
        if not shard.is_dir():
            continue
        for image_file in sorted(shard.iterdir()):
            if image_file.is_file() and image_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                return str(image_file)

    return ""


def _resolve_image_path(images_root: Path, fallback_image_path: str, article_id: str) -> str:
    if len(article_id) >= 3:
        shard = article_id[:3]
        for ext in (".jpg", ".jpeg", ".png"):
            image_file = images_root / shard / f"{article_id}{ext}"
            if image_file.exists():
                return str(image_file)

    return fallback_image_path


@st.cache_data(show_spinner=False)
def _load_article_lookup() -> Dict[str, Dict]:
    data_root = PROJECT_ROOT / "data"
    articles_csv = data_root / "articles.csv"
    images_root = data_root / "images"

    if not articles_csv.exists():
        return {}

    fallback_image_path = _discover_fallback_image_path(images_root)
    lookup: Dict[str, Dict] = {}

    with articles_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            article_id = (row.get("article_id") or "").strip()
            if not article_id:
                continue

            lookup[article_id] = {
                "article_id": article_id,
                "product_name": (row.get("prod_name") or "").strip() or f"Article {article_id}",
                "product_type_name": (row.get("product_type_name") or "").strip() or "Unknown",
                "colour_group_name": (row.get("colour_group_name") or "").strip() or "Unknown",
                "department_name": (row.get("department_name") or "").strip() or "Unknown",
                "garment_group_name": (row.get("garment_group_name") or "").strip() or "Unknown",
                "detail_desc": (row.get("detail_desc") or "").strip(),
                "image_path": _resolve_image_path(images_root, fallback_image_path, article_id),
                "price": _derive_price(article_id),
            }

    return lookup


def _coalesce_text(*values: Optional[object], default: str) -> str:
    for value in values:
        if isinstance(value, str):
            normalized = value.strip()
            if normalized:
                return normalized
        elif value is not None:
            return str(value)
    return default


def _coalesce_price(*values: Optional[object]) -> Optional[float]:
    for value in values:
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _coalesce_image_path(payload: Dict, base: Dict) -> str:
    for source in (payload, base):
        for key in ("image_path", "image_url"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _hydrate_item(item: Dict, article_lookup: Dict[str, Dict]) -> Dict:
    payload = dict(item)
    article_id = str(payload.get("article_id") or "").strip()
    base = article_lookup.get(article_id, {})

    normalized_id = article_id or str(base.get("article_id") or "unknown")
    payload["article_id"] = normalized_id
    payload["product_name"] = _coalesce_text(
        payload.get("product_name"),
        base.get("product_name"),
        default=f"Article {normalized_id}",
    )
    payload["product_type_name"] = _coalesce_text(
        payload.get("product_type_name"),
        base.get("product_type_name"),
        default="Unknown",
    )
    payload["colour_group_name"] = _coalesce_text(
        payload.get("colour_group_name"),
        base.get("colour_group_name"),
        default="Unknown",
    )
    payload["department_name"] = _coalesce_text(
        payload.get("department_name"),
        base.get("department_name"),
        default="Unknown",
    )
    payload["garment_group_name"] = _coalesce_text(
        payload.get("garment_group_name"),
        base.get("garment_group_name"),
        default="Unknown",
    )
    payload["detail_desc"] = _coalesce_text(
        payload.get("detail_desc"),
        base.get("detail_desc"),
        default="",
    )

    image_path = _coalesce_image_path(payload, base)
    if image_path:
        payload["image_path"] = image_path
        payload["image_url"] = image_path

    price = _coalesce_price(payload.get("price"), base.get("price"))
    if price is not None:
        payload["price"] = price

    try:
        payload["score"] = float(payload.get("score", 0.0))
    except (TypeError, ValueError):
        payload["score"] = 0.0

    return payload


def _client_signature(settings: SidebarSettings) -> str:
    mode = "stub" if settings.use_stub else "api"
    return f"{mode}:{settings.api_base_url}"


def _get_api_client(settings: SidebarSettings):
    signature = _client_signature(settings)
    if (
        st.session_state.get("api_client") is None
        or st.session_state.get("client_signature") != signature
    ):
        st.session_state.api_client = build_api_client(
            use_stub=settings.use_stub,
            api_base_url=settings.api_base_url,
        )
        st.session_state.client_signature = signature
    return st.session_state.api_client


def _encode_image(image_bytes):
    if not image_bytes:
        return None
    return base64.b64encode(image_bytes).decode("ascii")


def _normalize_search_results(results: List[Dict]) -> List[Dict]:
    article_lookup = _load_article_lookup()
    normalized = []
    for item in results:
        normalized.append(_hydrate_item(item, article_lookup))
    return normalized


def _normalize_recommendations(items: List[Dict]) -> List[Dict]:
    article_lookup = _load_article_lookup()
    normalized = []
    for item in items:
        normalized.append(_hydrate_item(item, article_lookup))
    return normalized


def _refresh_user_state(api_client, user_id: str) -> None:
    state = api_client.user_state(user_id=user_id)
    st.session_state.user_state = state
    st.session_state.interaction_count = int(state.get("interaction_count", 0))


def _run_search(api_client, settings: SidebarSettings, query: str, image_bytes) -> None:
    response = api_client.search(
        query=query or None,
        image_b64=_encode_image(image_bytes),
        filters=settings.filters,
        user_id=settings.user_id,
        alpha=settings.alpha,
        mode=settings.search_mode,
        limit=settings.search_limit,
    )
    results = _normalize_search_results(response.get("results", []))
    st.session_state.search_results = results
    st.session_state.search_feedback = (
        f"Mode: {response.get('query_mode', settings.search_mode)} | "
        f"Displayed: {len(results)} | Total matched: {response.get('total_results', len(results))}"
    )


def _run_recommend(api_client, settings: SidebarSettings, article_id: str) -> None:
    response = api_client.recommend(
        article_id=article_id,
        user_id=settings.user_id,
        k=settings.recommend_k,
    )
    st.session_state.recommendations = _normalize_recommendations(
        response.get("recommendations", [])
    )


def _run_before_after(api_client, settings: SidebarSettings, article_id: str) -> None:
    generic = api_client.recommend(article_id=article_id, user_id=None, k=settings.recommend_k)
    personalized = api_client.recommend(
        article_id=article_id,
        user_id=settings.user_id,
        k=settings.recommend_k,
    )
    st.session_state.comparison = {
        "generic": _normalize_recommendations(generic.get("recommendations", [])),
        "personalized": _normalize_recommendations(
            personalized.get("recommendations", [])
        ),
    }


def _run_interaction(api_client, settings: SidebarSettings, article_id: str) -> None:
    shown_articles = [
        str(item.get("article_id")) for item in st.session_state.get("search_results", [])
    ]
    response = api_client.interact(
        user_id=settings.user_id,
        article_id=article_id,
        shown_articles=shown_articles,
    )

    st.session_state.last_interaction = response
    st.session_state.interaction_count = int(response.get("interaction_count", 0))
    st.session_state.interaction_history.append(
        {
            "article_id": article_id,
            "adapted": bool(response.get("adapted", False)),
            "latency_ms": response.get("adaptation_time_ms"),
        }
    )
    st.session_state.interaction_history = st.session_state.interaction_history[-30:]

    _refresh_user_state(api_client, settings.user_id)
    _run_recommend(api_client, settings, article_id)


def main() -> None:
    init_session_state()
    inject_global_styles()
    render_hero()

    settings = render_sidebar()
    st.session_state.user_id = settings.user_id

    if settings.reset_clicked:
        reset_runtime_state()
        st.rerun()

    api_client = _get_api_client(settings)
    try:
        _refresh_user_state(api_client, settings.user_id)
    except APIClientError as exc:
        st.warning(f"User state endpoint unavailable: {exc}")

    query, image_bytes, submitted = render_search_form()

    if submitted:
        try:
            _run_search(api_client, settings, query, image_bytes)
        except APIClientError as exc:
            st.error(f"Search request failed: {exc}")

    render_feedback(st.session_state.get("search_feedback", ""))

    if st.session_state.search_results:
        st.subheader("Search results")
        selected = render_product_grid(st.session_state.search_results, columns=4)
        if selected is not None:
            st.session_state.selected_product = selected
            try:
                _run_recommend(api_client, settings, str(selected.get("article_id")))
                if settings.show_before_after:
                    _run_before_after(api_client, settings, str(selected.get("article_id")))
            except APIClientError as exc:
                st.error(f"Recommendation request failed: {exc}")

    selected_product = st.session_state.get("selected_product")
    if selected_product:
        st.markdown("---")
        render_selected_product(selected_product)

        action_col1, action_col2 = st.columns(2)
        with action_col1:
            if st.button("Record interaction", use_container_width=True):
                try:
                    _run_interaction(
                        api_client,
                        settings,
                        article_id=str(selected_product.get("article_id")),
                    )
                except APIClientError as exc:
                    st.error(f"Interaction request failed: {exc}")

        with action_col2:
            if st.button("Refresh recommendations", use_container_width=True):
                try:
                    _run_recommend(
                        api_client,
                        settings,
                        article_id=str(selected_product.get("article_id")),
                    )
                    if settings.show_before_after:
                        _run_before_after(
                            api_client,
                            settings,
                            article_id=str(selected_product.get("article_id")),
                        )
                except APIClientError as exc:
                    st.error(f"Refresh failed: {exc}")

        clicked_rec = render_recommendations(
            "You might also like",
            st.session_state.get("recommendations", []),
            columns=4,
        )

        if clicked_rec is not None:
            st.session_state.selected_product = {
                **selected_product,
                **clicked_rec,
            }
            try:
                _run_recommend(api_client, settings, str(clicked_rec.get("article_id")))
                if settings.show_before_after:
                    _run_before_after(api_client, settings, str(clicked_rec.get("article_id")))
            except APIClientError as exc:
                st.error(f"Could not load recommendation seed: {exc}")

    if settings.show_before_after and selected_product:
        comparison = st.session_state.get("comparison", {})
        render_before_after(
            comparison.get("generic", []),
            comparison.get("personalized", []),
        )

    st.markdown("---")
    render_personalization_panel(
        st.session_state.get("user_state"),
        st.session_state.get("last_interaction"),
    )

    with st.expander("Recent interactions", expanded=False):
        history = st.session_state.get("interaction_history", [])
        if not history:
            st.caption("No interactions yet.")
        else:
            st.dataframe(history, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
