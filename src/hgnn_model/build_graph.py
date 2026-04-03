import torch
import pandas as pd
from collections import defaultdict


def build_graph_hgnn(
    df,
    session_gap_days=7,
    max_session_len=20,
    window_size=3
):
    df = df.copy()
    df["t_dat"] = pd.to_datetime(df["t_dat"])

    weight_map = {
        "click": 1.0,
        "cart": 3.0,
        "purchase": 5.0
    }

    priority = {
        "click": 1,
        "cart": 2,
        "purchase": 3
    }

    edges_dict = {
        "click": [],
        "cart": [],
        "purchase": []
    }

    weights_dict = {
        "click": [],
        "cart": [],
        "purchase": []
    }

    # ===== BUILD SESSION GRAPH =====
    for user, user_df in df.groupby("customer_id"):
        user_df = user_df.sort_values("t_dat")

        items = user_df["article_id"].tolist()
        times = user_df["t_dat"].tolist()
        events = user_df["event_type"].tolist()

        if len(items) < 2:
            continue

        session = [(items[0], events[0])]

        for i in range(1, len(items)):
            gap = (times[i] - times[i - 1]).days

            if gap <= session_gap_days:
                session.append((items[i], events[i]))
            else:
                _add_edges_multi(session, edges_dict, weights_dict,
                                 weight_map, priority,
                                 max_session_len, window_size)
                session = [(items[i], events[i])]

        _add_edges_multi(session, edges_dict, weights_dict,
                         weight_map, priority,
                         max_session_len, window_size)

    # ===== NODE INDEX =====
    all_items = set()
    for rel in edges_dict:
        for a, b in edges_dict[rel]:
            all_items.add(a)
            all_items.add(b)

    id2idx = {aid: i for i, aid in enumerate(all_items)}

    edge_index_dict = {}
    edge_weight_dict = {}

    # ===== BUILD TENSOR =====
    for rel in edges_dict:
        edge_map = defaultdict(float)

        for (a, b), w in zip(edges_dict[rel], weights_dict[rel]):
            edge_map[(a, b)] += w

        edge_index = []
        edge_weight = []

        for (a, b), w in edge_map.items():
            if a == b:
                continue

            edge_index.append([id2idx[a], id2idx[b]])
            edge_weight.append(torch.log1p(torch.tensor(w)).item())

        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)

        # normalize weight
        edge_weight = edge_weight / (edge_weight.mean() + 1e-8)

        edge_index_dict[rel] = edge_index
        edge_weight_dict[rel] = edge_weight

        print(f"{rel}: {edge_index.shape[1]} edges")

    print("Num nodes:", len(id2idx))

    return edge_index_dict, edge_weight_dict, id2idx


def _add_edges_multi(
    session,
    edges_dict,
    weights_dict,
    weight_map,
    priority,
    max_session_len,
    window_size
):
    session = session[-max_session_len:]

    for i in range(len(session)):
        a, type_a = session[i]

        for j in range(i + 1, min(i + 1 + window_size, len(session))):
            b, type_b = session[j]

            rel = type_a if priority[type_a] >= priority[type_b] else type_b

            w = weight_map[type_a] + weight_map[type_b]

            edges_dict[rel].append((a, b))
            weights_dict[rel].append(w)

            edges_dict[rel].append((b, a))
            weights_dict[rel].append(w)