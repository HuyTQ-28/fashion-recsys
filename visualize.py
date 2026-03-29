import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# load
data = torch.load("data/article_embeddings_hgnn3.pt")
emb = data["embeddings"]
id2idx = data["id2idx"]

articles = pd.read_csv("data/articles.csv")
articles["article_id"] = articles["article_id"].astype(str).str.zfill(10)

idx2id = {v: str(k) for k, v in id2idx.items()}

# chọn top 10 category
top_labels = articles["product_type_name"].value_counts().head(10).index

N = 2000
indices = torch.randperm(len(emb))[:N]

emb_sample = emb[indices].detach().cpu().numpy()

labels = []
for i in indices:
    aid = idx2id[i.item()]
    row = articles[articles["article_id"] == aid]
    
    if len(row) > 0:
        label = row.iloc[0]["product_type_name"]
        if label in top_labels:
            labels.append(label)
        else:
            labels.append("Other")
    else:
        labels.append("Unknown")

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
emb_2d = tsne.fit_transform(emb_sample)

plt.figure(figsize=(10, 8))

for label in set(labels):
    idxs = [i for i, l in enumerate(labels) if l == label]
    plt.scatter(emb_2d[idxs, 0], emb_2d[idxs, 1], label=label, s=10)

plt.legend()
plt.title("t-SNE (Top Categories Only)")
plt.show()