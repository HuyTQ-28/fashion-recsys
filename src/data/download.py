"""
H&M Dataset Download & Subset Extraction.

Owner: Member 1 (Data & Graph Learning)

Usage:
    python -m src.data.download --output_dir data/raw --subset_date 2020-09-15
"""

import os
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def download_hm_dataset(output_dir: str = "data/raw") -> None:
    """
    Download the H&M Personalized Fashion Recommendations dataset from Kaggle.

    Requires:
        - Kaggle API credentials (~/.kaggle/kaggle.json or KAGGLE_USERNAME + KAGGLE_KEY env vars)

    Downloads:
        - articles.csv (~105K articles with metadata)
        - customers.csv (~1.3M customers)
        - transactions_train.csv (~31M transactions)
        - images/ (product images)

    Args:
        output_dir: Directory to save downloaded files.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    os.makedirs(output_dir, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    competition = "h-and-m-personalized-fashion-recommendations"
    logger.info(f"Downloading dataset from Kaggle competition: {competition}")
    api.competition_download_files(competition, path=output_dir, quiet=False)

    logger.info(f"Dataset downloaded to {output_dir}")


def extract_poc_subset(
    raw_dir: str = "data/raw",
    output_dir: str = "data/subset",
    target_date: str = "2020-09-15",
    min_user_interactions: int = 3,
) -> dict:
    """
    Extract a POC subset from the full H&M dataset.

    Selects transactions from a single day and the associated articles/customers.

    Args:
        raw_dir: Directory containing raw CSV files.
        output_dir: Directory to save the subset.
        target_date: Date string (YYYY-MM-DD) to extract.
        min_user_interactions: Minimum interactions per user to include.

    Returns:
        dict with counts: {n_transactions, n_articles, n_customers}
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load transactions
    logger.info("Loading transactions...")
    transactions = pd.read_csv(
        os.path.join(raw_dir, "transactions_train.csv"),
        dtype={"article_id": str, "customer_id": str},
        parse_dates=["t_dat"],
    )

    # Filter by date
    subset_txn = transactions[transactions["t_dat"] == target_date].copy()
    logger.info(f"Transactions on {target_date}: {len(subset_txn)}")

    # Filter users with minimum interactions
    user_counts = subset_txn["customer_id"].value_counts()
    active_users = user_counts[user_counts >= min_user_interactions].index
    subset_txn = subset_txn[subset_txn["customer_id"].isin(active_users)]

    # Get unique article and customer IDs
    article_ids = subset_txn["article_id"].unique()
    customer_ids = subset_txn["customer_id"].unique()

    # Load and filter articles
    articles = pd.read_csv(
        os.path.join(raw_dir, "articles.csv"),
        dtype={"article_id": str},
    )
    subset_articles = articles[articles["article_id"].isin(article_ids)]

    # Load and filter customers
    customers = pd.read_csv(
        os.path.join(raw_dir, "customers.csv"),
        dtype={"customer_id": str},
    )
    subset_customers = customers[customers["customer_id"].isin(customer_ids)]

    # Save subset
    subset_txn.to_csv(os.path.join(output_dir, "transactions.csv"), index=False)
    subset_articles.to_csv(os.path.join(output_dir, "articles.csv"), index=False)
    subset_customers.to_csv(os.path.join(output_dir, "customers.csv"), index=False)

    stats = {
        "n_transactions": len(subset_txn),
        "n_articles": len(subset_articles),
        "n_customers": len(subset_customers),
    }
    logger.info(f"POC subset saved to {output_dir}: {stats}")
    return stats


def extract_full_week(
    raw_dir: str = "data/raw",
    output_dir: str = "data/full_week",
    start_date: str = "2020-09-07",
    end_date: str = "2020-09-13",
) -> dict:
    """
    Extract a full week of transactions for Phase 2 training.

    Args:
        raw_dir: Directory containing raw CSV files.
        output_dir: Directory to save the full week data.
        start_date: Start date (inclusive).
        end_date: End date (inclusive).

    Returns:
        dict with counts.
    """
    os.makedirs(output_dir, exist_ok=True)

    transactions = pd.read_csv(
        os.path.join(raw_dir, "transactions_train.csv"),
        dtype={"article_id": str, "customer_id": str},
        parse_dates=["t_dat"],
    )

    mask = (transactions["t_dat"] >= start_date) & (transactions["t_dat"] <= end_date)
    subset = transactions[mask].copy()

    article_ids = subset["article_id"].unique()

    articles = pd.read_csv(
        os.path.join(raw_dir, "articles.csv"),
        dtype={"article_id": str},
    )
    subset_articles = articles[articles["article_id"].isin(article_ids)]

    subset.to_csv(os.path.join(output_dir, "transactions.csv"), index=False)
    subset_articles.to_csv(os.path.join(output_dir, "articles.csv"), index=False)

    stats = {
        "n_transactions": len(subset),
        "n_articles": len(subset_articles),
        "date_range": f"{start_date} to {end_date}",
    }
    logger.info(f"Full week data saved to {output_dir}: {stats}")
    return stats


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Download and prepare H&M dataset")
    parser.add_argument("--output_dir", default="data/raw", help="Raw data output directory")
    parser.add_argument("--subset_date", default="2020-09-15", help="POC subset date")
    parser.add_argument("--skip_download", action="store_true", help="Skip Kaggle download")
    args = parser.parse_args()

    if not args.skip_download:
        download_hm_dataset(args.output_dir)

    extract_poc_subset(args.output_dir, target_date=args.subset_date)
