import torch
import pandas as pd

data = torch.load("data/article_embeddings2.pt")
emb = data["embeddings"]
id2idx = data["id2idx"]

articles = pd.read_csv("data/articles.csv")
articles["article_id"] = articles["article_id"].astype(str).str.zfill(10)

idx2id = {v: str(k) for k, v in id2idx.items()}

def show_neighbors(article_id, topk=5):
    idx = id2idx[str(article_id)]
    query = emb[idx]

    dist = ((emb - query)**2).sum(dim=1)
    topk_idx = torch.topk(-dist, k=topk).indices

    print("\n🔍 Query:")
    print(articles[articles["article_id"] == str(article_id)][["prod_name", "product_type_name", "product_group_name"]])

    print("\n🔥 Neighbors:")
    for i in topk_idx:
        aid = idx2id[i.item()]
        print(articles[articles["article_id"] == aid][["prod_name", "product_type_name", "product_group_name"]])
        
        
show_neighbors("0108775044")
