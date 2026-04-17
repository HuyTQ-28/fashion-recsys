import pandas as pd
import numpy as np
import argparse
import logging
import os
import gc

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def shift_time_minutes(t_epoch: np.ndarray, min_m: int, max_m: int) -> np.ndarray:
    """Subtract random minutes between min_m and max_m (Causality Shift)."""
    shift_seconds = np.random.randint(min_m * 60, max_m * 60 + 1, size=len(t_epoch))
    return t_epoch - shift_seconds


def randomize_time_days(t_epoch: np.ndarray, min_global: int, max_global: int) -> np.ndarray:
    """Shift by a random number of days (-30 to +30), keeping hour-of-day intact, clamped to dataset bounds."""
    day_shifts = np.random.randint(-30, 31, size=len(t_epoch))
    seconds_shift = day_shifts * 86400
    new_t = t_epoch + seconds_shift
    return np.clip(new_t, min_global, max_global)


def main():
    parser = argparse.ArgumentParser(description="Generate scalable synthetic funnel data.")
    parser.add_argument("--input", default="dataset/subset_1week/transactions_train.csv", help="Original H&M transactions.")
    parser.add_argument("--output", default="dataset/subset_1week/synthetic_interactions.csv", help="Output file path.")
    parser.add_argument("--sample", action="store_true", help="Run on a 1%% sample for memory-safe testing.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # ---------------------------------------------------------
    # 0. Load Base Data (Memory Optimized)
    # ---------------------------------------------------------
    logger.info(f"Loading data from {args.input} (Sample Mode: {args.sample})...")
    
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}. Please provide a valid file to process.")
        return

    # Load only the 3 strictly necessary columns to save gigabytes of RAM
    usecols = ['customer_id', 'article_id', 't_dat']
    df = pd.read_csv(args.input, usecols=usecols)
    
    if args.sample:
        df = df.sample(frac=0.01, random_state=42).reset_index(drop=True)
        logger.info(f"Sampled 1%: Base size is {len(df)} rows.")

    df['event_type'] = 'purchase'

    # Convert t_dat to epoch (seconds) for ultra-fast np math
    logger.info("Converting timestamps to epoch (int64)...")
    # // 10**9 converts nanoseconds to seconds robustly across pandas versions
    df['t_epoch'] = pd.to_datetime(df['t_dat']).astype('int64') // 10**9 
    
    min_epoch = df['t_epoch'].min()
    max_epoch = df['t_epoch'].max()
    
    # Drop string t_dat to clear memory, we will convert back at the very end
    df.drop(columns=['t_dat'], inplace=True)
    P = len(df)
    logger.info(f"Base Purchases (P) = {P:,}. Target total events: ~{P*20:,}")

    # Reset output file
    if os.path.exists(args.output):
         os.remove(args.output)
    
    # Formatter Function
    def format_and_write(chunk_df: pd.DataFrame, mode='a', header=False):
        """Re-format the timestamp into string and append directly to the target CSV."""
        chunk_df['t_dat'] = pd.to_datetime(chunk_df['t_epoch'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
        out_cols = ['customer_id', 'article_id', 't_dat', 'event_type']
        chunk_df[out_cols].to_csv(args.output, mode=mode, header=header, index=False)
        gc.collect()

    # ---------------------------------------------------------
    # 1. Causal Rule (Purchases, Causal Carts, Causal Clicks)
    # ---------------------------------------------------------
    logger.info(f"Writing Phase 1: Original Purchases (1x, N={P:,})...")
    format_and_write(df.copy(), mode='w', header=True)
    
    logger.info(f"Writing Phase 1: Causal Carts [1-5 min prior] (1x, N={P:,})...")
    carts_causal = df.copy()
    carts_causal['event_type'] = 'cart'
    carts_causal['t_epoch'] = shift_time_minutes(carts_causal['t_epoch'].values, 1, 5)
    format_and_write(carts_causal.copy())

    logger.info(f"Writing Phase 1: Causal Clicks [1-3 min prior to cart] (1x, N={P:,})...")
    clicks_causal = carts_causal.copy()
    clicks_causal['event_type'] = 'click'
    clicks_causal['t_epoch'] = shift_time_minutes(clicks_causal['t_epoch'].values, 1, 3)
    format_and_write(clicks_causal)
    
    del carts_causal, clicks_causal
    gc.collect()

    # Chunk Generator
    def generate_random_block(target_event: str, multiplier: int):
        """Generator to yield N sets of P rows without exploding memory."""
        for i in range(multiplier):
            logger.info(f"  -> Generated {target_event} block {i+1}/{multiplier}...")
            # Real users, real distribution, randomized time bounds
            block = df.sample(n=P, replace=True).copy()
            block['event_type'] = target_event
            block['t_epoch'] = randomize_time_days(block['t_epoch'].values, min_epoch, max_epoch)
            yield block

    # ---------------------------------------------------------
    # 2. Abandoned Carts Rule (3P Carts + 3P preceding Clicks)
    # ---------------------------------------------------------
    logger.info("Writing Phase 2: Abandoned Sequence Phase (6x P total)...")
    for cart_block in generate_random_block('cart', 3):
        # 1. Abandoned cart
        format_and_write(cart_block.copy())
        
        # 2. Corresponding click 1-3 mins before the cart
        click_block = cart_block
        click_block['event_type'] = 'click'
        click_block['t_epoch'] = shift_time_minutes(click_block['t_epoch'].values, 1, 3)
        format_and_write(click_block)

    # ---------------------------------------------------------
    # 3. Window Shopping Rule (9P Clicks)
    # ---------------------------------------------------------
    logger.info("Writing Phase 3: Window Clicks Phase (9x P)...")
    for click_block in generate_random_block('click', 9):
        format_and_write(click_block)

    # ---------------------------------------------------------
    # 4. Window Shopping Rule (2P Favorites)
    # ---------------------------------------------------------
    logger.info("Writing Phase 4: Window Favorites Phase (2x P)...")
    for fav_block in generate_random_block('favorite', 2):
        format_and_write(fav_block)

    logger.info("Data Generation Complete! Distribution smoothly emulates real traffic.")
    logger.info(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()