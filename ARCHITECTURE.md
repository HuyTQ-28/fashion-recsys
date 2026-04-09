# Fashion Recommendation System — Kiến trúc & Luồng code

## Tổng quan hệ thống

Hệ thống gợi ý thời trang gồm 3 thành phần chính, mỗi thành phần do một member phụ trách:

```
┌─────────────────────────────────────────────────────────────────┐
│  M1 — Graph Learning       M2 — Personalization    M3 — Infra   │
│  (HGNN Teacher +           (Per-user MLP +         (Search +    │
│   Student MLP)              EMA + Triplet)          API + Redis) │
└─────────────────────────────────────────────────────────────────┘
         │                         │                       │
         ▼                         ▼                       ▼
  HGNN embeddings         Personal MLP per user    Weaviate / Redis
  Student MLP ckpt        UserState (EMA vector)   REST endpoints
  [N, 64] per article     Lifecycle Manager        Hybrid search
```

---

## 1. Cây thư mục và mục đích từng file

```
src/
├── data/
│   ├── download.py              # Tải H&M dataset (Kaggle API), trích subset
│   ├── build_graph.py           # Xây co-purchase graph → HeteroData (PyG)
│   └── extract_embeddings.py   # FashionCLIP → Tensor[512] cho từng article
│
├── models/
│   ├── hgnn.py                  # Teacher: HGNN 512→256→128→64 (SAGEConv)
│   ├── student_mlp.py           # Student: MLP 512→256→128→64 (knowledge distill)
│   ├── personal_mlp.py          # Per-user MLP (deep copy student, adapt per user)
│   └── mock_student_mlp.py     # Mock dùng cho unit test
│
├── training/
│   ├── train_hgnn.py            # Training loop HGNN (contrastive loss)
│   ├── train_student.py         # Distillation loop (MSE alignment loss)
│   └── adapt_personal_mlp.py   # Triplet loss adaptation (per user, online)
│
├── inference/
│   ├── user_state.py            # EMA vector per user, serialization
│   ├── personal_mlp.py          # [xem models/] factory + lifecycle entry
│   ├── mlp_lifecycle.py         # LRU cache + pluggable backend (Redis/Dict)
│   ├── recommender.py           # 2-stage recall+rerank, PersonalizationEngine API
│   ├── session_simulator.py     # Replay lịch sử giao dịch để đánh giá
│   └── redis_client.py          # Upstash Redis backend (StorageBackend protocol)
│
├── search/
│   ├── clip_encoder.py          # FashionCLIP encode text/image → Tensor[512]
│   ├── weaviate_setup.py        # Định nghĩa schema 2 collection Weaviate
│   ├── ingest_products.py       # Đẩy embeddings + metadata vào Weaviate
│   └── search_engine.py         # Hybrid search (BM25 + vector, Weaviate)
│
├── evaluation/
│   ├── metrics.py               # P@K, R@K, F1@K
│   └── sensitivity.py           # Sweep hyperparameter, tìm best config
│
└── modal_app/
    └── app.py                   # Template deploy lên Modal (serverless GPU)
```

---

## 2. Luồng xử lý tổng thể

```
Dữ liệu thô (H&M)
      │
      ▼
[OFFLINE TRAINING — M1]
      │
      ├─► extract_embeddings.py  →  clip_emb[article_id] = Tensor[512]
      │
      ├─► build_graph.py         →  HeteroData graph (node=articles, edge=co-purchase)
      │
      ├─► train_hgnn.py          →  HGNN embeddings  hgnn_emb[article_id] = Tensor[64]
      │                                               + model checkpoint
      └─► train_student.py       →  Student MLP checkpoint  (512→256→128→64)
                                     student_emb[article_id] = Tensor[64]
                                     ↓
[PERSONALIZATION ENGINE — M2]
      │
      ├─► personal_mlp.py        →  Per-user MLP (deep copy student per user)
      ├─► user_state.py          →  EMA vector[64] per user
      ├─► adapt_personal_mlp.py  →  Triplet loss update weights (online)
      └─► mlp_lifecycle.py       →  LRU cache + Redis storage
                                     ↓
[SEARCH & API — M3]
      │
      ├─► ingest_products.py     →  CLIP + MLP embeddings → Weaviate
      ├─► search_engine.py       →  Hybrid search endpoint
      ├─► recommender.py         →  2-stage personalized recommendation
      └─► redis_client.py        →  Lưu Personal MLP (TTL 14 ngày)
                                     ↓
[EVALUATION]
      ├─► session_simulator.py   →  Replay giao dịch, tạo ground truth
      ├─► metrics.py             →  P@10, R@10, F1@10
      └─► sensitivity.py         →  Sweep alpha/margin/steps/... → best config
```

---

## 3. Chi tiết từng bước — đầu vào / đầu ra

### 3.1 Trích xuất CLIP embeddings

**File:** [src/data/extract_embeddings.py](src/data/extract_embeddings.py)

| | |
|---|---|
| **Input** | Ảnh sản phẩm `images/{prefix}/{article_id}.jpg` |
| **Process** | FashionCLIP vision encoder, batch=32, chuẩn hóa L2 |
| **Output** | `Dict[article_id → Tensor[512]]` (L2-normalized) |

---

### 3.2 Xây dựng co-purchase graph

**File:** [src/data/build_graph.py](src/data/build_graph.py)

| | |
|---|---|
| **Input** | `transactions.csv` + `clip_embeddings` |
| **Process** | Group by customer + time window (24h), đếm co-purchase pairs |
| **Output** | `HeteroData` với node features `[N, 512]`, edge indices, edge weights |

```
transactions.csv
  customer_A: [article_1, article_2, article_3]  # cùng ngày
              → edge (article_1, article_2) weight=1
              → edge (article_1, article_3) weight=1
              → edge (article_2, article_3) weight=1
```

---

### 3.3 Training HGNN (Teacher)

**File:** [src/training/train_hgnn.py](src/training/train_hgnn.py)

| | |
|---|---|
| **Input** | `HeteroData` (node features `[N, 512]`, edges, weights) |
| **Architecture** | `SAGEConv: 512→256→128→64` (per relation, mean aggregation) |
| **Loss** | Contrastive: `Σ w·‖h_p − h_q‖² (pos) − max(0, margin − ‖h_neg‖²)` |
| **Optimizer** | Adam (lr=1e-3), early stopping patience=5 |
| **Output** | HGNN checkpoint + `Dict[article_id → Tensor[64]]` (structural embeddings) |

```
Node features [N, 512] ──┐
                          ├─► HGNNLayer(512→256) ─► ReLU
Edge indices ─────────────┤
                          ├─► HGNNLayer(256→128) ─► ReLU
Edge weights ─────────────┘
                              HGNNLayer(128→64)
                                    │
                                    ▼
                          Structural embeddings [N, 64]
```

---

### 3.4 Training Student MLP (Knowledge Distillation)

**File:** [src/training/train_student.py](src/training/train_student.py)

| | |
|---|---|
| **Input** | `clip_embeddings [N, 512]` + `hgnn_embeddings [N, 64]` (frozen teacher) |
| **Architecture** | `Linear(512→256) → ReLU → Linear(256→128) → ReLU → Linear(128→64)` |
| **Loss** | `MSE(StudentMLP(clip), HGNN(clip).detach())` |
| **Optimizer** | Adam (lr=1e-3), batch=1024, early stopping patience=5 |
| **Output** | Student MLP checkpoint (`mlp_student.pt`) |

```
clip[512] ──► StudentMLP ──► pred[64]
                               │
hgnn[64] (frozen) ─────────────►  MSE Loss
```

**Tại sao cần Student?** Không cần load graph lúc serving — Student chỉ cần CLIP embedding của article là predict được embedding 64-dim.

---

### 3.5 Tạo Personal MLP

**File:** [src/models/personal_mlp.py](src/models/personal_mlp.py)

| | |
|---|---|
| **Input** | Student MLP checkpoint + `user_id` |
| **Process** | `PersonalMLPFactory.create(user_id)` → `deepcopy` Student MLP |
| **Output** | `PersonalMLP` object (512→256→128→64, weights = Student MLP) |
| **Size** | ~700 KB khi serialize (base64 state_dict) |

Mỗi user có một bản sao riêng của Student MLP. Bản sao này được update liên tục theo hành vi.

---

### 3.6 Cập nhật EMA (Exponential Moving Average)

**File:** [src/inference/user_state.py](src/inference/user_state.py)

| | |
|---|---|
| **Input** | `mlp_proj [64]` (embedding của article qua Personal MLP) + `alpha=0.7` |
| **Process** | `u_t = (1−α)·u_{t−1} + α·mlp_proj` |
| **Output** | `ema_vector [64]` (đại diện sở thích hiện tại của user) |

```
Tương tác 1: u = h_1                    (khởi tạo lần đầu)
Tương tác 2: u = 0.3·h_1 + 0.7·h_2     (alpha=0.7: nghiêng về recent)
Tương tác 3: u = 0.3·u_2 + 0.7·h_3
...
```

`alpha=0.7` (optimal từ sensitivity) nghĩa là sở thích gần đây chiếm 70%, quá khứ 30%.

---

### 3.7 Triplet Loss Adaptation

**File:** [src/training/adapt_personal_mlp.py](src/training/adapt_personal_mlp.py)

Trigger mỗi khi tích lũy đủ **9 tương tác** (`trigger_every_n=9`, optimal).

| | |
|---|---|
| **Input** | `personal_mlp`, `positive_clips [B, 512]`, `negative_clips [B, 512]` |
| **Process** | 1 SGD step (optimal) minimize `dist(proj_wc, proj_pos)` |
| **Output** | `personal_mlp` weights cập nhật in-place |

```
Bước 1: h_wc = weighted_centroid(positive_clips)   [512]
        w_purchase=4, w_click=1, w_view=1

Bước 2: proj_wc  = personal_mlp(h_wc)              [1,   64]
        proj_pos = personal_mlp(h_pos)              [B,   64]
        proj_neg = personal_mlp(h_neg)              [B,   64]

Bước 3: loss = mean(‖proj_wc − proj_pos‖²)
        (margin=∞: bỏ qua negatives — optimal từ sensitivity)

Bước 4: loss.backward() + SGD step (lr=1e-4)
```

**Tại sao margin=∞?** Trên fake data, chỉ cần kéo positives lại gần anchor. Không cần đẩy negatives ra xa. Đây là kết quả tốt nhất từ sensitivity sweep (F1 +1.80 so với margin=1.0).

---

### 3.8 MLP Lifecycle Manager (Cache + Storage)

**File:** [src/inference/mlp_lifecycle.py](src/inference/mlp_lifecycle.py)

| | |
|---|---|
| **Hot tier** | LRU in-memory cache, max 200 users (~140 MB) |
| **Cold tier** | Redis (Upstash) hoặc `LocalDictBackend` (test) |
| **TTL** | 14 ngày (user không tương tác → xóa khỏi Redis) |

```
get_or_create(user_id)
        │
        ├─ Hit LRU cache?    ──► return CacheEntry  (O(1))
        │
        ├─ Hit Redis?        ──► deserialize PersonalMLP → insert LRU
        │
        └─ New user          ──► factory.create(user_id)
                                  UserState(alpha=0.7)
                                  insert LRU → evict oldest if full
                                  flush evicted to Redis if dirty
```

**CacheEntry** chứa:
- `personal_mlp`: `PersonalMLP` object
- `user_state`: EMA vector + interaction count
- `interaction_batch`: list article_id đang tích lũy
- `dirty`: flag cần flush Redis

---

### 3.9 Sinh gợi ý (2-stage Recall + Re-rank)

**File:** [src/inference/recommender.py](src/inference/recommender.py)

#### Cold user (chưa có tương tác)

```
seed_article_id
      │
      ▼
clip_embeddings[seed]         [512]
      │
      ▼
cdist(seed, all_clips)        [N]
      │
      ▼
Top-K article IDs (item-to-item KNN)
```

#### Warm user (có EMA vector)

```
ema_vector [64]  +  personal_mlp
      │
      ├─── STAGE 1: RECALL (top-100) ──────────────────────────┐
      │    For each article:                                     │
      │      clip_emb [512] → personal_mlp → proj [64]          │
      │    cdist(ema, proj_all) → sort → top-100 candidates     │
      │                                                          │
      └─── STAGE 2: RE-RANK (top-10) ──────────────────────────┘
           For each candidate:
             clip_emb [512] → personal_mlp → proj [64]
           cdist(ema, proj_candidates) → sort → top-10
                 │
                 ▼
           List[article_id] (10 items)
```

---

### 3.10 Xử lý tương tác người dùng

**File:** [src/inference/recommender.py](src/inference/recommender.py) — `PersonalizationEngine.handle_interaction()`

```
Input: user_id, article_id, interaction_type, shown_articles

Step 1: Get entry = lifecycle.get_or_create(user_id)

Step 2: clip_emb = clip_embeddings[article_id]           [512]
        mlp_proj = personal_mlp(clip_emb)                [64]

Step 3: user_state.update_ema(mlp_proj)
        ema_vector = 0.3·ema_old + 0.7·mlp_proj

Step 4: interaction_batch.append(article_id)

Step 5: If len(batch) >= 9:
          positives = [clip_embeddings[aid] for aid in batch]
          negatives = shown_articles không bị click, hoặc random
          TripletAdaptation.adapt(personal_mlp, pos, neg)
          lifecycle.mark_dirty(user_id)
          lifecycle.flush(user_id)     ← serialize → Redis
          batch = []

Output: {status, interaction_count, adapted, adaptation_time_ms}
```

---

### 3.11 Hybrid Search

**File:** [src/search/search_engine.py](src/search/search_engine.py)

| | |
|---|---|
| **Input** | Text query hoặc image → `CLIPEncoder` → `query_vec [512]` |
| **Process** | Weaviate hybrid: `α·vector_score + (1−α)·BM25_score`, α=0.7 |
| **Output** | `List[Dict{article_id, product_name, score, metadata}]` (top-20) |

---

### 3.12 Session Simulation (Evaluation)

**File:** [src/inference/session_simulator.py](src/inference/session_simulator.py)

```
test_transactions.csv
      │
      ▼
Nhóm theo user, sort theo t_dat
      │
      For each user:
        For each interaction (article_i, t_i):
          [1] lifecycle.get_or_create(user_id)
          [2] update EMA
          [3] adapt nếu đủ batch
          [4] ground_truth = articles[i+1 : i+13]  # 12 bước tiếp theo
          [5] recommendations = KNN từ ema + personal_mlp
          [6] Tính P@10, R@10, F1@10
      │
      ▼
Dict[user_id → List[records]]
```

---

### 3.13 Metrics

**File:** [src/evaluation/metrics.py](src/evaluation/metrics.py)

```
recs  = [r1, r2, ..., r10]          # 10 gợi ý (ordered)
gt    = [g1, g2, ..., g12]          # 12 articles mua tiếp theo

P@10  = |recs ∩ gt| / 10
R@10  = |recs ∩ gt| / |gt|
F1@10 = 2·P·R / (P + R)

Báo cáo × 10^4 theo quy ước bài báo.
```

---

## 4. Hyperparameters tối ưu (từ Sensitivity Sweep)

Sweep chạy trên **200 users**, **fake_behavior.csv**, **HGNN 64-dim embeddings**, **Colab T4 GPU**.

| Hyperparameter | Giá trị sweep | **Best** | Default cũ | Delta F1 |
|---|---|---|---|---|
| `alpha` (EMA) | 0.1, 0.3, 0.5, 0.7, 1.0 | **0.7** | 0.5 | +0.80 |
| `sgd_steps` | 1, 5, 20, 50 | **1** | 5 | +1.21 |
| `margin` | 0.1, 1.0, 10.0, ∞ | **∞** | 1.0 | +1.80 |
| `batch_size` | 1, 3, 5, 9 | **9** | 5 | +0.96 |
| `lifespan_days` | 1, 7, 14, 21 | 1–21 (bằng nhau) | 14 | 0.00 |

**Best overall config:** F1@10 = 48.66 × 10⁻⁴ (tại margin=∞)

**Insight:**
- `margin=∞`: chỉ minimize dist(anchor, positives) là đủ — negatives không cần thiết trên fake data
- `sgd_steps=1`: 1 bước SGD là tối ưu, thêm bước chỉ overfit
- `alpha=0.7`: học nhanh từ hành vi gần đây
- `lifespan_days`: không ảnh hưởng vì fake data không có temporal gap thực

---

## 5. Sơ đồ phụ thuộc giữa các module

```
                    ┌─────────────────────────────┐
                    │  M1: Data & Graph Learning   │
                    │  extract_embeddings.py        │
                    │  build_graph.py               │
                    │  train_hgnn.py                │
                    │  train_student.py             │
                    └──────────────┬──────────────┘
                                   │ Student MLP ckpt
                                   │ clip_embeddings
                                   │ hgnn_embeddings
                      ┌────────────┴────────────┐
                      ▼                         ▼
          ┌───────────────────────┐   ┌──────────────────────┐
          │ M2: Personalization   │   │ M3: Search & Infra   │
          │ personal_mlp.py       │◄──│ ingest_products.py   │
          │ user_state.py         │   │ search_engine.py     │
          │ adapt_personal_mlp.py │   │ weaviate_setup.py    │
          │ mlp_lifecycle.py      │──►│ redis_client.py      │
          │ recommender.py        │   │ clip_encoder.py      │
          └───────────┬───────────┘   └──────────────────────┘
                      │
                      ▼
          ┌───────────────────────┐
          │ Evaluation            │
          │ session_simulator.py  │
          │ metrics.py            │
          │ sensitivity.py        │
          └───────────────────────┘
```

**Chiều dữ liệu:**
- M1 → M2: `mlp_student.pt`, `article_embeddings_hgnn3.pt`
- M1 → M3: CLIP embeddings, MLP embeddings, product metadata
- M2 → M3: Personal MLP serialized (Redis), UserState
- M3 → M2: `StorageBackend` (Redis implementation)
- M2 ↔ Eval: `PersonalizationEngine`, `SessionSimulator`

---

## 6. Input/Output tóm tắt theo từng bước

| Bước | File | Input | Output |
|---|---|---|---|
| Extract CLIP | `extract_embeddings.py` | Ảnh `[H×W×3]` | `Dict[id → Tensor[512]]` |
| Build graph | `build_graph.py` | Transactions + CLIP | `HeteroData` |
| Train HGNN | `train_hgnn.py` | HeteroData | HGNN ckpt + `Dict[id → Tensor[64]]` |
| Train Student | `train_student.py` | CLIP + HGNN | Student ckpt (`mlp_student.pt`) |
| Create PersonalMLP | `personal_mlp.py` | Student ckpt + `user_id` | `PersonalMLP` (deepcopy, `[512→64]`) |
| Update EMA | `user_state.py` | `mlp_proj[64]`, `alpha=0.7` | `ema_vector[64]` |
| Triplet Adapt | `adapt_personal_mlp.py` | MLP + pos/neg clips `[B,512]` | MLP weights in-place |
| Cache/Load | `mlp_lifecycle.py` | `user_id` | `CacheEntry{mlp, state, batch}` |
| Recall (Stage 1) | `recommender.py` | `ema[64]` + `personal_mlp` | Top-100 `article_id` |
| Re-rank (Stage 2) | `recommender.py` | Top-100 + `ema[64]` | Top-10 `article_id` |
| Search | `search_engine.py` | Text/image query | `List[{id, name, score}]` |
| Simulate | `session_simulator.py` | Transactions CSV | `Dict[user → List[records]]` |
| Metrics | `metrics.py` | recs + ground_truth | `{P@10, R@10, F1@10}` × 10⁴ |

---

## 7. Artifacts chính

| File | Mô tả | Kích thước |
|---|---|---|
| `src/models/article_embeddings_hgnn3.pt` | HGNN embeddings cho 19,278 articles | ~5 MB |
| `src/models/mlp_student.pt` | Student MLP weights | ~700 KB |
| `data/fake_behavior.csv` | Fake behavior data (72K users, 2.1M rows) | ~150 MB |
| Redis `mlp:{user_id}` | Personal MLP per user (base64) | ~700 KB/user |
