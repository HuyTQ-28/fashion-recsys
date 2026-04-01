"""
Streamlit Frontend — Fashion Recommender System.

Owner: Member 4 (Frontend & Evaluation)

Search-first UX flow:
1. User types a query or uploads an image
2. Results displayed as a product grid with metadata
3. Click a product -> shows "You might also like" recommendations
4. Each click adapts the Personal MLP in real-time
"""

import streamlit as st

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Fashion Recommender",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Sidebar — Filters & Settings
# ============================================================

def render_sidebar():
    """Render metadata filter sidebar."""
    st.sidebar.title("🔍 Search Filters")

    # Search mode
    search_mode = st.sidebar.selectbox(
        "Search Mode",
        ["Hybrid", "Semantic", "Keyword"],
        index=0,
    )

    # Hybrid alpha (only shown for hybrid mode)
    alpha = 0.7
    if search_mode == "Hybrid":
        alpha = st.sidebar.slider(
            "Hybrid Alpha (0=BM25, 1=Vector)",
            0.0, 1.0, 0.7, 0.1,
        )

    # Metadata filters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Metadata Filters")

    colour = st.sidebar.selectbox("Color", ["All", "Black", "White", "Blue", "Red", "Green", "Pink", "Grey"])
    department = st.sidebar.selectbox("Department", ["All", "Ladieswear", "Menswear", "Divided", "Kids"])
    product_type = st.sidebar.text_input("Product Type", placeholder="e.g., T-shirt, Dress")

    # Build filters dict
    filters = {}
    if colour != "All":
        filters["colour_group_name"] = colour
    if department != "All":
        filters["department_name"] = department
    if product_type:
        filters["product_type_name"] = product_type

    # Personalization info
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Personalization")
    if "interaction_count" in st.session_state:
        st.sidebar.metric("Interactions", st.session_state.interaction_count)
        st.sidebar.metric("Cache Status", st.session_state.get("cache_status", "N/A"))
    else:
        st.sidebar.info("No interactions yet")

    return search_mode.lower(), alpha, filters


# ============================================================
# Main Content
# ============================================================

def render_search_bar():
    """Render the search bar and image upload."""
    col1, col2 = st.columns([3, 1])

    with col1:
        query = st.text_input(
            "🔍 Search for fashion products",
            placeholder="e.g., red summer dress, men's denim jacket...",
            key="search_query",
        )

    with col2:
        uploaded_image = st.file_uploader("📸 Or upload an image", type=["jpg", "jpeg", "png"])

    return query, uploaded_image


def render_product_grid(results: list, cols: int = 5):
    """Render search results as a product grid."""
    if not results:
        st.info("No results found. Try a different search query.")
        return

    for i in range(0, len(results), cols):
        row = st.columns(cols)
        for j, col in enumerate(row):
            idx = i + j
            if idx < len(results):
                item = results[idx]
                with col:
                    # Product card
                    st.markdown(f"**{item.get('product_name', 'Unknown')}**")
                    st.caption(f"{item.get('colour_group_name', '')} • {item.get('product_type_name', '')}")
                    st.caption(f"Score: {item.get('score', 0):.3f}")

                    if st.button("View", key=f"product_{idx}"):
                        st.session_state.selected_product = item
                        st.session_state.interaction_count = st.session_state.get("interaction_count", 0) + 1


def render_recommendations(recommendations: list):
    """Render the 'You might also like' panel."""
    if not recommendations:
        return

    st.markdown("---")
    st.subheader("✨ You Might Also Like")

    cols = st.columns(5)
    for i, rec in enumerate(recommendations[:5]):
        with cols[i]:
            st.markdown(f"**{rec.get('product_name', 'Unknown')}**")
            st.caption(f"Score: {rec.get('score', 0):.3f}")


def render_comparison_panel(generic_results: list, personalized_results: list):
    """Render before/after comparison panel."""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔲 Generic Results")
        for item in generic_results[:5]:
            st.text(f"• {item.get('product_name', 'Unknown')}")

    with col2:
        st.subheader("⭐ Personalized Results")
        for item in personalized_results[:5]:
            st.text(f"• {item.get('product_name', 'Unknown')}")


# ============================================================
# Main App
# ============================================================

def main():
    st.title("👗 Fashion Recommender System")
    st.caption("Real-time personalized recommendations powered by HGNN + FashionCLIP")

    # Initialize session state
    if "interaction_count" not in st.session_state:
        st.session_state.interaction_count = 0
    if "user_id" not in st.session_state:
        st.session_state.user_id = "demo_user"

    # Sidebar
    search_mode, alpha, filters = render_sidebar()

    # Search bar
    query, uploaded_image = render_search_bar()

    # Search
    if query or uploaded_image:
        st.markdown("---")

        # TODO: Call Modal API /search endpoint
        # For now, show placeholder
        st.subheader(f"🔍 Results for: {query or 'Image search'}")
        st.info(
            "Connect to Modal API to see real results. "
            "Run: `modal deploy src/modal_app/app.py`"
        )

        # Placeholder results (replace with actual API call)
        mock_results = [
            {
                "article_id": f"0{i:08d}",
                "product_name": f"Sample Product {i}",
                "product_type_name": "T-shirt",
                "colour_group_name": "Blue",
                "score": 0.95 - i * 0.05,
            }
            for i in range(10)
        ]
        render_product_grid(mock_results)

    # Show recommendations if a product is selected
    if "selected_product" in st.session_state:
        product = st.session_state.selected_product
        st.markdown("---")
        st.subheader(f"📄 Selected: {product.get('product_name', '')}")

        # TODO: Call Modal API /recommend and /interact endpoints
        render_recommendations([])

    # Show comparison toggle
    show_comparison = st.sidebar.checkbox("Show Before/After Comparison")
    if show_comparison:
        render_comparison_panel([], [])


if __name__ == "__main__":
    main()
