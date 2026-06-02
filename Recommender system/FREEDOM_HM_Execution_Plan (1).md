# FREEDOM × H&M — End-to-End Execution Plan

**Author role:** Senior ML Engineer (Graph Recommender Systems)
**Goal:** Train, evaluate, and qualitatively validate the FREEDOM multimodal recommender on the H&M Personalized Fashion Recommendation dataset, with **minimal modification** to the official repository [`enoche/FREEDOM`](https://github.com/enoche/FREEDOM) (which is built on the **MMRec** framework).

---

## 0. Repository & Framework Anatomy (Read This First)

FREEDOM ships as a single model plugged into the **MMRec** codebase. The relevant directory tree:

```
FREEDOM/
├── src/
│   ├── main.py                       # entry point
│   ├── configs/
│   │   ├── overall.yaml              # global defaults (epochs, lr, device, etc.)
│   │   ├── model/FREEDOM.yaml        # FREEDOM hyperparameters (k, α_v, λ, dropout…)
│   │   └── dataset/{name}.yaml       # per-dataset config — we will CREATE hm.yaml
│   ├── models/freedom.py             # frozen kNN graph + denoised UI graph + LightGCN
│   ├── common/
│   │   ├── trainer.py                # training loop, evaluation, early stopping
│   │   └── abstract_recommender.py
│   ├── utils/
│   │   ├── dataset.py                # loads .inter + feat.npy files
│   │   └── data_utils.py
│   └── data/                         # <-- expected dataset root
│       └── {name}/
│           ├── {name}.inter          # user-item interactions (TSV)
│           ├── image_feat.npy        # (N_items, D_v) float32
│           └── text_feat.npy         # (N_items, D_t) float32
```

**Key constraint:** MMRec identifies items by a **contiguous integer ID** (0..N-1) and users likewise. Every downstream artifact (the `.npy` feature matrices, the `.inter` file, the kNN graph) is indexed by this internal ID. **Maintain a single source-of-truth mapping `article_id ↔ item_idx` and `customer_id ↔ user_idx`** — this is the single most common source of silent bugs.

---

## 1. Optimal Data Sampling Strategy

### 1.1 Why random / full-history sampling fails on H&M

H&M `transactions_train.csv` spans **~2 years (Sep 2018 → Sep 2020)** with ~31M rows. Naïvely using everything — or sampling uniformly at random — breaks the assumptions FREEDOM relies on:

| Failure mode | Mechanism | Consequence for FREEDOM |
|---|---|---|
| **Seasonal contamination** | A user who bought a wool coat in Jan 2019 and a swimsuit in Jul 2019 appears as a single user node connecting items that are *never co-purchasable in reality*. | Pollutes the user–item graph; LightGCN propagation smears summer↔winter signals; kNN item graph picks up spurious cross-season "neighbors". |
| **Fast-fashion drift / cold items** | Many `article_id`s appear for one short drop window and never again. | Items have degree 1–3 across the full history but degree 0 in any test window → metrics dominated by unrecommendable items. |
| **Catalog churn vs. fixed N** | FREEDOM freezes a kNN item-item graph of size N×N. If N is bloated with discontinued SKUs, the graph wastes capacity and inflates memory. | Quadratic kNN build cost on ~100k+ items; most edges connect dead inventory. |
| **Test-period mismatch** | The Kaggle competition's hidden test is **the week immediately following** the training cutoff. | A model trained on uniformly-sampled history doesn't see recent fashion trends and underperforms on MAP@12. |
| **Graph connectivity** | Random row sampling produces a fragmented bipartite graph with many isolated components. | LightGCN message passing degenerates — embeddings of disconnected components don't co-evolve. |

### 1.2 Recommended strategy: **time-windowed sliding cut**

**Use a contiguous 6–8 week window ending at the dataset's last date** (2020-09-22). This single decision resolves all five failure modes above.

| Choice | Rationale |
|---|---|
| **Window length: 8 weeks** (≈ 56 days, 2020-07-29 → 2020-09-22) | Long enough to give each *active* item ≥5 interactions for K-core filtering; short enough to keep one season (late summer → early autumn) coherent. Empirically, 6 weeks is too tight for K-core on most users; >10 weeks starts dragging in mid-summer items. |
| **Chronological split** train/val/test = **last 7 days as test, the 7 days before that as val, the remaining ~6 weeks as train** | Mirrors the Kaggle evaluation protocol (predict next week). FREEDOM's BPR loss assumes test data is *unseen future*; random splits leak information. |
| **No negative downsampling on transactions** | Each (customer_id, article_id, t_dat) row becomes one positive edge. Duplicates (same user-item bought twice) → collapsed to a single edge with implicit weight 1. |
| **Stratify the val/test cut by user** | Within the 7-day val/test windows, only keep users with ≥2 interactions in that window so leave-one-out is meaningful. |

**Concrete cutoff dates:**

```python
TRAIN_END = "2020-09-08"   # inclusive
VAL_END   = "2020-09-15"
TEST_END  = "2020-09-22"   # dataset max
WINDOW_START = "2020-07-29"  # 8 weeks before TEST_END
```

### 1.3 Expected scale after windowing

Order-of-magnitude estimates (will vary):

- Transactions in window: ~3–4M
- Unique customers: ~600k–900k
- Unique articles: ~40k–60k

After 5-core filtering (Section 2.3), expect ~150k–300k users × ~25k–35k items × ~2–3M edges — comparable to MMRec's `clothing` and `sports` benchmarks. **This is the right operating regime for FREEDOM.**

---

## 2. Multimodal Data Preprocessing Pipeline

### 2.1 Visual features

**Recommendation: FashionCLIP (`patrickjohncyh/fashion-clip`) image tower.**

- **Why not vanilla ResNet-50?** ResNet trained on ImageNet captures generic object semantics; it is weak at distinguishing "boyfriend jeans" vs. "skinny jeans" or fine color/pattern attributes that drive fashion co-purchase.
- **Why not OpenAI CLIP ViT-B/32?** Acceptable fallback, but FashionCLIP is post-trained on 800k fashion image-caption pairs and consistently beats CLIP on H&M-like retrieval tasks.
- **Output dim:** 512 (matches FREEDOM's expectations after the visual MLP projection).

**Image lookup:** images live at `images/{first_3_digits_of_article_id}/{article_id}.jpg`.

```python
def img_path(article_id: int, root="images/"):
    s = f"{article_id:010d}"  # zero-pad to 10 digits
    return os.path.join(root, s[:3], f"{s}.jpg")
```

**Handling missing images:** ~5% of articles lack images.
- **Do not** drop them — many are textually rich (e.g., accessories).
- Fill with the **mean visual embedding** computed over articles *that have images*. This is the standard MMRec convention and keeps the `image_feat.npy` matrix aligned with the item index. (FREEDOM's `image_dropout` will partially mask these during training.)
- Track which items were filled in a boolean mask file `missing_image_mask.npy` for later diagnostic use.

**Batched extraction sketch:**

```python
import torch, numpy as np
from transformers import CLIPProcessor, CLIPModel
model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").eval().cuda()
proc  = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

# items_df is ordered by item_idx (0..N-1)
feats = np.zeros((len(items_df), 512), dtype=np.float32)
missing = np.zeros(len(items_df), dtype=bool)
BATCH = 128
with torch.no_grad():
    for i in range(0, len(items_df), BATCH):
        chunk = items_df.iloc[i:i+BATCH]
        imgs, idxs = [], []
        for j, row in chunk.iterrows():
            p = img_path(row.article_id)
            if os.path.exists(p):
                imgs.append(Image.open(p).convert("RGB"))
                idxs.append(row.item_idx)
            else:
                missing[row.item_idx] = True
        if imgs:
            inputs = proc(images=imgs, return_tensors="pt").to("cuda")
            emb = model.get_image_features(**inputs).cpu().numpy()
            feats[idxs] = emb
# fill missing with mean of present
feats[missing] = feats[~missing].mean(axis=0, keepdims=True)
np.save("image_feat.npy", feats)
np.save("missing_image_mask.npy", missing)
```

### 2.2 Textual features

**Recommendation: FashionCLIP text tower** (so visual & text live in the same pretraining manifold — improves the cosine-similarity step of the frozen item-item graph).

Alternative: **Sentence-BERT `all-mpnet-base-v2`** (768-dim, stronger pure semantics but mismatched space). If you want strict alignment with the FREEDOM paper, pick S-BERT; if you want the best practical H&M result, pick FashionCLIP.

**Text construction (concatenate per article):**

```python
def build_text(row):
    parts = [
        row.prod_name,
        row.product_type_name,
        row.product_group_name,
        row.graphical_appearance_name,
        row.colour_group_name,
        row.perceived_colour_master_name,
        row.department_name,
        row.index_name,
        str(row.detail_desc) if pd.notna(row.detail_desc) else "",
    ]
    return ". ".join(p for p in parts if p and p != "nan")
```

This template surfaces both **categorical structure** (type, color, department) and **free-form description**, which empirically gives the best downstream NDCG.

Save as `text_feat.npy` with shape `(N_items, D_t)`, dtype `float32`, **L2-normalize each row** (matches FREEDOM's cosine-similarity assumption).

### 2.3 K-core filtering for the user-item bipartite graph

FREEDOM (and MMRec generally) expect a **5-core** graph: every user and every item has ≥5 interactions. Apply iteratively to convergence:

```python
def k_core(df, k=5, user_col="customer_id", item_col="article_id"):
    while True:
        before = len(df)
        uc = df[user_col].value_counts()
        ic = df[item_col].value_counts()
        df = df[df[user_col].isin(uc[uc >= k].index) &
                df[item_col].isin(ic[ic >= k].index)]
        if len(df) == before:
            return df
```

**Order of operations is critical:**

1. Window-filter transactions to `[WINDOW_START, TEST_END]`.
2. Deduplicate (customer_id, article_id) pairs — keep first `t_dat` for chronological splitting.
3. **Apply 5-core on the train portion only.**
4. Remap to contiguous `user_idx` and `item_idx` (0..N-1).
5. Filter val/test to keep only users and items present in train. (This is standard inductive-on-IDs hygiene; FREEDOM is transductive over IDs.)
6. Extract image/text features **in item_idx order**.

### 2.4 Final files produced

```
data/hm/
├── hm.inter              # TSV: userID \t itemID \t timestamp  (train+val+test, with flag column)
├── image_feat.npy        # (N_items, 512) float32, row i ↔ item_idx i
├── text_feat.npy         # (N_items, 512 or 768) float32
├── id_map_user.csv       # customer_id, user_idx
├── id_map_item.csv       # article_id, item_idx, has_image (bool)
└── splits/
    ├── train.tsv
    ├── val.tsv
    └── test.tsv
```

The `.inter` file header expected by MMRec is:

```
userID:token	itemID:token	x_label:float
```

where `x_label` is a split indicator (0=train, 1=val, 2=test). Some MMRec forks use separate files; the FREEDOM repo's README documents the exact convention — **check `src/utils/dataset.py` against your version** before writing.

---

## 3. Adapting the GitHub Repository

The whole point of this exercise is **minimal code change**. You will modify exactly three things and write zero new model code.

### 3.1 Drop in the dataset

Place the artifacts from §2.4 at `FREEDOM/src/data/hm/`. MMRec discovers datasets by folder name.

### 3.2 Create `configs/dataset/hm.yaml`

Copy the existing `baby.yaml` or `clothing.yaml` as a starting point and edit:

```yaml
# configs/dataset/hm.yaml
field_separator: "\t"
seq_separator: " "

USER_ID_FIELD: userID
ITEM_ID_FIELD: itemID
TIME_FIELD: timestamp

inter_file_name: hm.inter
user_graph_dict_file: user_graph_dict.npy   # built on first run
vision_feature_file: image_feat.npy
text_feature_file: text_feat.npy

# splits: explicit because we did chronological splitting
split_mode: x_label                # MMRec convention; reads the split column
load_col:
  inter: [userID, itemID, x_label]

# evaluation
metrics: ["Recall", "NDCG", "Precision", "MAP"]
topk: [10, 20, 50]
valid_metric: Recall@20
eval_batch_size: 4096
```

### 3.3 Tune `configs/model/FREEDOM.yaml`

Start from paper defaults and adjust two knobs for H&M scale:

```yaml
embedding_size: 64
n_layers: 2                  # LightGCN depth on user-item graph
n_ui_layers: 2
knn_k: 10                    # paper-recommended top-k for frozen item graph
mm_image_weight: 0.1         # α_v in S = α_v·S^v + (1-α_v)·S^t
                             # H&M textual features are very strong → lower α_v
dropout: 0.8                 # degree-sensitive edge pruning rate
reg_weight: [1e-4]
learning_rate: [1e-3]
lambda_coeff: 0.5            # reconstruction-loss weight λ
train_batch_size: 4096
epochs: 200
stopping_step: 20            # early stop patience
```

Why `mm_image_weight: 0.1` instead of the paper's 0.1–0.5? H&M product titles are exceptionally informative (`"Slim High Jeans, Black, Denim"`). Down-weighting vision typically helps unless your visual encoder is FashionCLIP (in which case 0.3–0.5 is competitive — sweep this).

### 3.4 Do you need to touch the data loader?

- **If you use MMRec's `.inter` + `x_label` split convention:** **no code changes**. The loader auto-handles it.
- **If your version uses separate `train.tsv/val.tsv/test.tsv`:** modify `src/utils/dataset.py` only in the `_load_split_data()` method to read three files — about 15 lines.
- **Graph construction scripts:** **do not touch.** `models/freedom.py` builds the frozen kNN graph from `image_feat.npy` and `text_feat.npy` on the first epoch and caches it. Letting it run is faster than reimplementing.

### 3.5 One gotcha to pre-empt

The frozen-graph construction inside `freedom.py` does an `O(N²)` similarity computation **once**. For N ≈ 35k this is ~5GB FP32 in RAM. If you hit OOM, patch the `build_knn_neighbourhood()` helper to compute similarities in chunks (~5k items at a time) — a 10-line change, no algorithmic difference.

---

## 4. Training & Evaluation Execution

### 4.1 Environment

```bash
git clone https://github.com/enoche/FREEDOM.git && cd FREEDOM
conda create -n freedom python=3.9 -y && conda activate freedom
pip install torch==2.0.1 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install transformers==4.36.0 sentence-transformers pandas pyarrow tqdm
```

### 4.2 Run preprocessing (your script from §2)

```bash
python preprocess_hm.py \
  --transactions ~/kaggle/hm/transactions_train.csv \
  --articles     ~/kaggle/hm/articles.csv \
  --images_root  ~/kaggle/hm/images \
  --window_start 2020-07-29 --test_end 2020-09-22 \
  --kcore 5 \
  --out src/data/hm
```

### 4.3 Train

The frozen item-item graph is built **lazily on the first epoch** and cached to disk — there is no separate "pre-training" command, despite what the FREEDOM paper's phrasing suggests. One command:

```bash
cd src
python main.py -m FREEDOM -d hm
```

To pin GPU and seed:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py -m FREEDOM -d hm --seed 2024
```

Hyperparameter sweep (the repo bundles a grid runner):

```bash
python main.py -m FREEDOM -d hm \
  --hyper_file configs/hyper/FREEDOM.yaml
```

Training logs go to `log/FREEDOM/hm/`; the best checkpoint is saved at `saved/FREEDOM-hm-best.pth`.

### 4.4 Standard metrics (Recall@20, NDCG@20)

These are emitted automatically by the trainer at every `eval_step`. The final eval block prints:

```
test result:
recall@10 : 0.0xxx | recall@20 : 0.0xxx | recall@50 : 0.0xxx
ndcg@10   : 0.0xxx | ndcg@20   : 0.0xxx | ndcg@50   : 0.0xxx
```

### 4.5 Producing H&M-style MAP@12 predictions

MAP@12 over **all 1.3M Kaggle customers** is not what the trainer reports (it reports leave-one-out style metrics on filtered users). To produce a Kaggle-style submission, write a short inference script:

```python
# infer_map12.py — run from src/
import torch, numpy as np, pandas as pd
from utils.dataset import RecDataset
from models.freedom import FREEDOM
from utils.configurator import Config

config = Config(model="FREEDOM", dataset="hm")
dataset = RecDataset(config)
model = FREEDOM(config, dataset).cuda()
model.load_state_dict(torch.load("saved/FREEDOM-hm-best.pth")["state_dict"])
model.eval()

with torch.no_grad():
    user_emb, item_emb = model.forward()       # h_u, h_i
    scores = user_emb @ item_emb.T              # (U, I)
    # mask training interactions
    scores[dataset.train_uid, dataset.train_iid] = -1e9
    topk = scores.topk(12, dim=1).indices.cpu().numpy()

# reverse-map to article_id and customer_id
item_map = pd.read_csv("data/hm/id_map_item.csv")
user_map = pd.read_csv("data/hm/id_map_user.csv")
idx2art  = item_map.set_index("item_idx")["article_id"].to_dict()
idx2cust = user_map.set_index("user_idx")["customer_id"].to_dict()

sub = pd.DataFrame({
    "customer_id": [idx2cust[i] for i in range(len(user_emb))],
    "prediction":  [" ".join(f"0{idx2art[j]}" for j in row) for row in topk],
})
sub.to_csv("submission.csv", index=False)
```

For the **~1M customers not in the windowed training set** (the cold-start tail Kaggle scores you on), append a popularity-based fallback prediction: top-12 most-purchased articles in the last 7 days. Concatenate to `submission.csv`.

To compute MAP@12 locally on your held-out test week, use the official competition formula on `topk` vs. `dataset.test_uid_iid` ground truth.

---

## 5. Qualitative Validation (Sanity Checks)

These are not optional. A model can post strong Recall@20 while having learned nonsensical latent geometry — these checks catch that.

### 5.1 Visual Nearest Neighbors in the learned space

The intent: confirm FREEDOM's final item embedding `h_i` captures **aesthetic and category coherence**, not just popularity bias.

```python
import torch, numpy as np, pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# h_i from §4.5, shape (N, D)
h_i = item_emb.cpu().numpy()
h_i_norm = h_i / np.linalg.norm(h_i, axis=1, keepdims=True)

articles = pd.read_csv("data/hm/id_map_item.csv").merge(
    pd.read_csv("~/kaggle/hm/articles.csv"), on="article_id")

def visual_neighbors(anchor_idx, k=10):
    sims = h_i_norm @ h_i_norm[anchor_idx]
    return np.argsort(-sims)[1:k+1]   # skip self

# pick a few anchors across categories
anchors = articles.groupby("product_group_name").sample(1, random_state=0).item_idx[:6]
fig, axes = plt.subplots(len(anchors), 11, figsize=(22, 2.2*len(anchors)))
for r, a in enumerate(anchors):
    nbrs = visual_neighbors(a, 10)
    for c, idx in enumerate([a, *nbrs]):
        aid = articles.loc[articles.item_idx==idx, "article_id"].iloc[0]
        p = f"images/{str(aid).zfill(10)[:3]}/{str(aid).zfill(10)}.jpg"
        axes[r, c].imshow(Image.open(p)) if os.path.exists(p) else axes[r, c].axis("off")
        axes[r, c].set_title("anchor" if c==0 else f"#{c}", fontsize=8)
        axes[r, c].axis("off")
plt.savefig("qual_visual_neighbors.png", dpi=120, bbox_inches="tight")
```

**What "good" looks like:** anchor = striped tee → neighbors are other tees, mostly striped, in adjacent colors. **What "bad" looks like:** anchor = striped tee → neighbors include random handbags or kids' shoes (= category collapse, often a symptom of α_v misconfiguration or BPR regularization being too high).

**Cross-check against the raw FashionCLIP space:** repeat the same script using raw `image_feat.npy` instead of `h_i`. FREEDOM's `h_i` should look *at least* as coherent — if it looks *worse*, your training has collapsed the multimodal signal and you need to raise `lambda_coeff`.

### 5.2 t-SNE / UMAP projection colored by `product_group_name`

```python
import umap, matplotlib.pyplot as plt, seaborn as sns
reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=0)
emb2d = reducer.fit_transform(h_i)

articles = articles.set_index("item_idx").sort_index()
plt.figure(figsize=(14, 12))
sns.scatterplot(
    x=emb2d[:,0], y=emb2d[:,1],
    hue=articles.product_group_name.values,
    s=4, alpha=0.6, palette="tab20", legend="brief",
)
plt.title("FREEDOM item embeddings (UMAP) — colored by product_group_name")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, markerscale=2)
plt.savefig("qual_umap_product_group.png", dpi=140, bbox_inches="tight")
```

**Acceptance criteria:**
- Major groups (`Garment Upper body`, `Garment Lower body`, `Shoes`, `Accessories`, `Underwear`) form **visually separable macro-clusters**.
- Adjacent groups (e.g., `Garment Upper body` ↔ `Garment Full body`) should *bleed into each other* — that's the model correctly capturing semantic adjacency, not failing.
- Watch for **"comet tail" structures** trailing into low-density regions: usually long-tail items with weak training signal. Acceptable, but if the tails dominate, raise the K-core threshold.

**Also recommended:** run the same UMAP colored by `colour_group_name` and by `index_name` (Ladieswear / Menswear / Divided / etc.). If the index_name plot does *not* show clear separation, the model has under-fit the demographic signal — usually fixable with another 50 epochs.

### 5.3 Quantitative tie-breaker

If §5.1 and §5.2 look ambiguous, compute a **category purity metric**:

```python
# for each item, fraction of its top-10 NN that share product_group_name
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=11, metric="cosine").fit(h_i)
_, idx = nn.kneighbors(h_i)
purity = np.mean([
    np.mean(articles.product_group_name.values[idx[i, 1:]] ==
            articles.product_group_name.values[i])
    for i in range(len(h_i))
])
print(f"Top-10 category purity: {purity:.3f}")
```

Healthy range on H&M: **0.55–0.75**. Above 0.85 means the model collapsed to category-only — multimodal signal is being wasted. Below 0.45 means the latent space is incoherent.

---

## 6. Suggested Order of Operations (Checklist)

1. ☐ Clone repo, set up env (§4.1).
2. ☐ Write `preprocess_hm.py` implementing §1.2 + §2.1–§2.3.
3. ☐ Verify `id_map_item.csv` ↔ `image_feat.npy` row alignment with a spot-check.
4. ☐ Create `configs/dataset/hm.yaml` (§3.2).
5. ☐ Run a **smoke test**: subset to 10k users / 5k items, train 5 epochs, confirm metrics > 0.
6. ☐ Full training run on the 8-week window (§4.3). Expect ~3–6 hrs on a single A100.
7. ☐ Generate Recall@20 / NDCG@20 from trainer logs.
8. ☐ Run `infer_map12.py` (§4.5), compute local MAP@12 on the held-out test week.
9. ☐ Visual NN sanity check (§5.1).
10. ☐ UMAP plot + category purity (§5.2–§5.3).
11. ☐ If purity ∈ [0.55, 0.75] and Recall@20 > 0.05 → ship. Otherwise sweep `mm_image_weight` and `lambda_coeff`.

---

## 7. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| OOM on `O(N²)` kNN graph build | Chunked similarity (§3.5); or temporarily raise K-core to 10. |
| FashionCLIP download blocked in your environment | Fallback to `openai/clip-vit-base-patch32`; minor metric hit. |
| Many users have only 1–2 interactions in window | Lower K-core to 3 on user side only; keep 5 on item side. |
| Trainer crashes on `x_label` split mode | Older MMRec uses `split_ratio`; switch to writing three separate files and patch `_load_split_data()` (§3.4). |
| Submission validation fails on Kaggle | Ensure `article_id` is zero-padded to 10 chars with leading `0`; this is the canonical Kaggle format. |

---

**Bottom line:** the FREEDOM repo is a near-drop-in solution for H&M *if* the data is sliced temporally (not randomly), if the kNN graph is built on a multimodal encoder aligned with the fashion domain (FashionCLIP), and if you respect the strict item-index alignment between `.inter`, `image_feat.npy`, and `text_feat.npy`. Everything else — model code, training loop, evaluation — can stay untouched.
