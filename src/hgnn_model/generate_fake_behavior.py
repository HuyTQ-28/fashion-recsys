import pandas as pd
import numpy as np
import random
import time
from tqdm import tqdm

tqdm.pandas()

def generate_fake_behavior(data_dir="dataset/subset_1week"):
    print("1. Loading and formatting data...")
    start_time = time.time()
    
    articles = pd.read_csv(f"{data_dir}/articles.csv", dtype={'article_id': str})
    transactions = pd.read_csv(f"{data_dir}/transactions_train.csv", dtype={'article_id': str, 'customer_id': str})

    articles["article_id"] = articles["article_id"].str.zfill(10)
    transactions["article_id"] = transactions["article_id"].str.zfill(10)
    transactions["t_dat"] = pd.to_datetime(transactions["t_dat"])

    # Build item similarity dictionary
    print("2. Building item similarity dictionary...")
    grouped_items = articles.groupby(["product_type_name", "product_group_name"])['article_id'].agg(list).to_dict()
    
    # Map article_id -> tuple(type, group)
    article_keys = zip(articles['product_type_name'], articles['product_group_name'])
    article_to_group = dict(zip(articles['article_id'], article_keys))

    sim_dict = {}
    for art_id, grp_tuple in tqdm(article_to_group.items(), desc="Building Sim Dict"):
        group_items = grouped_items.get(grp_tuple, [])

        sim_items = [i for i in group_items if i != art_id]
        if len(sim_items) > 50:
            sim_items = random.sample(sim_items, 50)
        sim_dict[art_id] = sim_items

    # Prepare purchase transactions
    print("3. Processing purchases...")
    purchases = transactions[['t_dat', 'customer_id', 'article_id']].copy()
    purchases['event_type'] = 'purchase'
    purchases['purchase_t_dat'] = purchases['t_dat']

    # Generate clicks
    print("4. Generating clicks...")
    def get_clicks(art_id):
        items = sim_dict.get(art_id, [])
        return random.sample(items, min(5, len(items))) if items else []

    # Apply only on article_id column
    purchases['click_items'] = purchases['article_id'].progress_apply(get_clicks)

    # Explode list of clicked items into separate rows
    clicks = purchases.explode('click_items').dropna(subset=['click_items']).copy()
    clicks = clicks.rename(columns={'click_items': 'clicked_article'})

    # Calculate time: T - (3 - i)
    # cumcount() helps count the order i (0,1,2,3,4) of clicks generated from the same purchase
    clicks['click_order'] = clicks.groupby(clicks.index).cumcount()
    clicks['t_dat'] = clicks['purchase_t_dat'] - pd.to_timedelta(3 - clicks['click_order'], unit='D')
    clicks['event_type'] = 'click'

    # Generate carts
    print("5. Generating carts (Vectorized)...")
    clicks['orig_prefix'] = clicks['article_id'].str[:3]
    clicks['click_prefix'] = clicks['clicked_article'].str[:3]

    # Calculate probability using Numpy instead of if/else
    prob_array = np.where(clicks['orig_prefix'] == clicks['click_prefix'], 0.6, 0.3)
    rand_array = np.random.rand(len(clicks))

    # Filter clicks to convert to carts
    carts = clicks[rand_array < prob_array].copy()
    carts['event_type'] = 'cart'
    # Cart happens 1 day before Purchase
    carts['t_dat'] = carts['purchase_t_dat'] - pd.Timedelta(days=1)

    # Clean up and merge
    print("6. Formatting and sorting...")
    clicks['article_id'] = clicks['clicked_article']
    carts['article_id'] = carts['clicked_article']

    cols = ['t_dat', 'customer_id', 'article_id', 'event_type']
    
    fake_df = pd.concat([
        purchases[cols], 
        clicks[cols], 
        carts[cols]
    ], ignore_index=True)

    # Sort by customer and time
    fake_df = fake_df.sort_values(["customer_id", "t_dat"])

    # Save result
    out_path = f"{data_dir}/fake_behavior.csv"
    fake_df.to_csv(out_path, index=False)
    
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f} seconds!")
    print(f"Data distribution:\n{fake_df['event_type'].value_counts()}")

if __name__ == "__main__":
    generate_fake_behavior()