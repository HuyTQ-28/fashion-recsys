import torch
import pandas as pd
import random
import torch.nn.functional as F

# ===== LOAD =====
data = torch.load("data/article_embeddings_hgnn3.pt")
emb = data["embeddings"]
id2idx = data["id2idx"]

# 🔥 normalize embedding (rất quan trọng)
emb = F.normalize(emb, dim=1)

articles = pd.read_csv("data/articles.csv")
articles["article_id"] = articles["article_id"].astype(str).str.zfill(10)

idx2id = {v: str(k) for k, v in id2idx.items()}

# ===== HELPER =====
def get_info(aid):
    row = articles[articles["article_id"] == aid]
    if len(row) == 0:
        return f"{aid} (NOT FOUND)"
    
    row = row.iloc[0]
    return f"{row['prod_name']} | {row['product_type_name']} | {row['product_group_name']}"

# ===== MAIN =====
def show_neighbors(article_id, topk=10):
    article_id = str(article_id)

    if article_id not in id2idx:
        print("❌ Article not found in embedding")
        return

    idx = id2idx[article_id]
    query = emb[idx]

    # 🔥 cosine similarity (tốt hơn L2 khi đã normalize)
    scores = torch.matmul(emb, query)

    # 🔥 lấy topk +1 để loại chính nó
    topk_idx = torch.topk(scores, k=topk + 1).indices

    print("\n🔍 Query:")
    print(get_info(article_id))

    print("\n🔥 Neighbors:")
    count = 0
    for i in topk_idx:
        aid = idx2id[i.item()]

        if aid == article_id:
            continue  # skip chính nó

        print(f"{count+1}. {get_info(aid)}")
        count += 1

        if count == topk:
            break


# ===== TEST =====
random_article_id = random.choice(list(id2idx.keys()))
print(f"\n🎯 Random product ID: {random_article_id}")

show_neighbors(random_article_id, topk=10)