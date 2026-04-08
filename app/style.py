"""Visual styling for the Streamlit frontend."""

import streamlit as st


def inject_global_styles() -> None:
    """Inject custom CSS for a more branded ecommerce look."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');

        :root {
            --bg: #f7f2e8;
            --bg-soft: #fff9ef;
            --card: #ffffff;
            --text: #1f2933;
            --muted: #5a6472;
            --accent: #c45a2d;
            --accent-2: #0f766e;
            --line: #eedfcb;
            --chip: #f3e4d0;
        }

        .stApp {
            background: radial-gradient(circle at 8% 12%, #ffe8c7 0%, transparent 35%),
                        radial-gradient(circle at 88% 18%, #d9f6ef 0%, transparent 30%),
                        linear-gradient(180deg, var(--bg-soft) 0%, var(--bg) 70%);
            color: var(--text);
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        h1, h2, h3, h4 {
            font-family: 'Space Grotesk', sans-serif !important;
            letter-spacing: -0.02em;
            color: var(--text);
        }

        .fr-hero {
            padding: 1.1rem 1.2rem;
            border-radius: 16px;
            border: 1px solid var(--line);
            background: linear-gradient(135deg, #fff9ef 0%, #ffffff 60%, #f5f8ff 100%);
            margin-bottom: 1rem;
            box-shadow: 0 6px 20px rgba(62, 40, 15, 0.05);
        }

        .fr-tag {
            display: inline-block;
            margin-right: 0.45rem;
            margin-top: 0.35rem;
            padding: 0.22rem 0.5rem;
            border-radius: 999px;
            background: var(--chip);
            color: #5a3d1f;
            font-size: 0.78rem;
            font-weight: 600;
        }

        .fr-card {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 0.75rem 0.8rem;
            min-height: 160px;
            box-shadow: 0 4px 14px rgba(40, 30, 15, 0.05);
            margin-bottom: 0.55rem;
        }

        .fr-card-title {
            font-size: 0.98rem;
            font-weight: 700;
            line-height: 1.3;
            margin-bottom: 0.35rem;
            color: var(--text);
        }

        .fr-meta {
            color: var(--muted);
            font-size: 0.8rem;
            line-height: 1.45;
        }

        .fr-score {
            margin-top: 0.45rem;
            color: var(--accent-2);
            font-size: 0.85rem;
            font-weight: 700;
        }

        .fr-selected {
            border-left: 4px solid var(--accent);
            background: #fffef9;
            border-radius: 8px;
            padding: 0.75rem 0.8rem;
            margin-bottom: 0.8rem;
        }

        .fr-caption {
            color: var(--muted);
            font-size: 0.86rem;
        }

        div[data-testid="metric-container"] {
            border: 1px solid var(--line);
            border-radius: 12px;
            background: #fffdfa;
            padding: 0.2rem 0.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
