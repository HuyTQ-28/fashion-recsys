import pandas as pd

def split_by_time(path):
    df = pd.read_csv(path)
    df["article_id"] = df["article_id"].astype(str).str.zfill(10)
    
    df["t_dat"] = pd.to_datetime(df["t_dat"])
    df = df.sort_values("t_dat")

    unique_days = sorted(df["t_dat"].dt.date.unique())

    train_days = unique_days[:5]
    val_day = unique_days[5]
    test_days = unique_days[6:]

    train_df = df[df["t_dat"].dt.date.isin(train_days)]
    val_df   = df[df["t_dat"].dt.date == val_day]
    test_df  = df[df["t_dat"].dt.date.isin(test_days)]

    print("Train days:", train_days)
    print("Val day:", val_day)
    print("Test days:", test_days)

    print("Train size:", len(train_df))
    print("Val size:", len(val_df))
    print("Test size:", len(test_df))

    return train_df, val_df, test_df