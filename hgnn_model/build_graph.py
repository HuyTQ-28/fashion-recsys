import torch
import pandas as pd


def build_graph(df, session_gap_days=1):
    df = df.copy()
    df["t_dat"] = pd.to_datetime(df["t_dat"])

    edges = []

    for user, user_df in df.groupby("customer_id"):
        user_df = user_df.sort_values("t_dat")

        items = user_df["article_id"].tolist()
        times = user_df["t_dat"].tolist()

        if len(items) < 2:
            continue

        session = [items[0]]

        for i in range(1, len(items)):
            gap = (times[i] - times[i-1]).days

            if gap <= session_gap_days:
                session.append(items[i])
            else:
                edges += build_session_edges(session)
                session = [items[i]]

        # session cuối
        edges += build_session_edges(session)

    # build id2idx
    all_items = set([a for e in edges for a in e])
    id2idx = {aid: i for i, aid in enumerate(all_items)}

    edge_index = torch.tensor(
        [[id2idx[a], id2idx[b]] for a, b in edges],
        dtype=torch.long
    ).t()

    print("Num nodes:", len(id2idx))
    print("Num edges:", edge_index.shape[1])

    return edge_index, id2idx


def build_session_edges(session):
    edges = []

    for i in range(len(session) - 1):
        a = session[i]
        b = session[i + 1]

        edges.append((a, b))
        edges.append((b, a))  # bidirectional

    return edges

def build_edges_only(df, id2idx, session_gap_days=1):
    df = df.copy()
    df["t_dat"] = pd.to_datetime(df["t_dat"])

    edges = []

    for user, user_df in df.groupby("customer_id"):
        user_df = user_df.sort_values("t_dat")

        items = user_df["article_id"].tolist()
        times = user_df["t_dat"].tolist()

        if len(items) < 2:
            continue

        session = [items[0]]

        for i in range(1, len(items)):
            gap = (times[i] - times[i-1]).days

            if gap <= session_gap_days:
                session.append(items[i])
            else:
                edges += build_session_edges_filtered(session, id2idx)
                session = [items[i]]

        edges += build_session_edges_filtered(session, id2idx)

    if len(edges) == 0:
        return None

    return torch.tensor(edges, dtype=torch.long).t()


def build_session_edges_filtered(session, id2idx):
    edges = []

    for i in range(len(session) - 1):
        a = session[i]
        b = session[i + 1]

        if a in id2idx and b in id2idx:
            edges.append((id2idx[a], id2idx[b]))
            edges.append((id2idx[b], id2idx[a]))

    return edges
