# M2 — Personalization Engine: Báo cáo & Hướng dẫn cho M3

---

## 1. Tổng quan M2 đã làm

M2 triển khai **Personalization Engine** — hệ thống cá nhân hóa gợi ý theo từng user dựa trên Knowledge Distillation + Triplet Loss (paper Section 2).

### Các file đã hoàn thiện

| File                                   | Mô tả                                                                                                |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `src/models/personal_mlp.py`         | `PersonalMLP` (deep-copy Student MLP per user), `PersonalMLPFactory`, serialize/deserialize base64 |
| `src/inference/user_state.py`        | `UserState` — EMA vector [64] per user (Eq. 5, α=0.7)                                              |
| `src/training/adapt_personal_mlp.py` | `TripletAdaptation` — SGD adaptation per user (Eq. 6-7)                                             |
| `src/inference/mlp_lifecycle.py`     | `MLPLifecycleManager` — LRU cache + pluggable `StorageBackend`                                    |
| `src/inference/recommender.py`       | `PersonalizationEngine` — API chính cho M3 gọi                                                    |
| `src/inference/session_simulator.py` | `SessionSimulator` — replay transactions để evaluate                                              |
| `src/evaluation/metrics.py`          | `precision_at_k`, `recall_at_k`, `f1_at_k`                                                       |
| `src/evaluation/sensitivity.py`      | Hyperparameter sweep (đã chạy trên Colab)                                                          |

### Hyperparameters tối ưu (từ sensitivity sweep, 200 users)

| Tham số                     | Giá trị tối ưu                    | F1 cải thiện |
| ---------------------------- | ------------------------------------- | -------------- |
| EMA alpha (α)               | **0.7**                         | +0.80          |
| SGD steps/adaptation         | **1**                           | +1.21          |
| Triplet margin (ε)          | **∞** (minimize dist_pos only) | +1.80          |
| Trigger every N interactions | **9**                           | +0.96          |

### Kết quả evaluation (CLIP 512-dim, 20 users, 50 interactions)

```
Precision@10 : 44.00 ± 48.41  (×10⁴)
Recall@10    : 37.14 ± 40.76  (×10⁴)
F1@10        : 40.26 ± 44.24  (×10⁴)
```

---

## 2. Artifacts bàn giao cho M3

| File                                       | Mô tả                                  | M3 dùng để làm gì                         |
| ------------------------------------------ | ---------------------------------------- | ---------------------------------------------- |
| `src/models/mlp_student.pt`              | Student MLP weights (512→256→128→64)  | Khởi tạo `PersonalMLPFactory`              |
| `src/models/clip_embeddings.pt`          | CLIP embeddings 512-dim, 19,284 articles | Feed vào `PersonalizationEngine`            |
| `src/models/article_embeddings_hgnn3.pt` | HGNN embeddings 64-dim + id2idx          | Ingest vào Weaviate `ProductRec` collection |

---

## 3. Hướng dẫn M3 tích hợp

### Bước 1 — Implement Redis backend

M2 đã định nghĩa interface `StorageBackend` trong `src/inference/mlp_lifecycle.py`. M3 chỉ cần implement bằng Upstash Redis thật:

```python
# src/inference/redis_client.py — M3 implement
from upstash_redis import Redis
from src.inference.mlp_lifecycle import StorageBackend

class UpstashRedisBackend:
    def __init__(self, url: str, token: str):
        self._client = Redis(url=url, token=token)

    def get(self, key: str):
        return self._client.get(key)

    def set(self, key: str, value: str, ttl_seconds: int):
        self._client.set(key, value, ex=ttl_seconds)

    def delete(self, key: str):
        self._client.delete(key)

    def exists(self, key: str) -> bool:
        return self._client.exists(key) > 0

    def refresh_ttl(self, key: str, ttl_seconds: int):
        self._client.expire(key, ttl_seconds)
```

### Bước 2 — Khởi tạo PersonalizationEngine trong Modal app

```python
# src/modal_app/app.py — M3 khởi tạo
import torch
from src.models.personal_mlp import PersonalMLPFactory
from src.inference.mlp_lifecycle import MLPLifecycleManager
from src.inference.recommender import PersonalizationEngine
from src.training.adapt_personal_mlp import TripletAdaptation
from src.inference.redis_client import UpstashRedisBackend  # M3 implement

# Load artifacts
clip_embeddings = torch.load("src/models/clip_embeddings.pt", weights_only=True)

# Khởi tạo factory từ Student MLP
factory = PersonalMLPFactory("src/models/mlp_student.pt")

# Redis backend (M3 cung cấp credentials)
backend = UpstashRedisBackend(
    url=os.environ["UPSTASH_REDIS_URL"],
    token=os.environ["UPSTASH_REDIS_TOKEN"],
)

# Lifecycle manager
lifecycle = MLPLifecycleManager(
    factory=factory,
    backend=backend,
    max_size=200,       # LRU cache 200 users ~140MB
    alpha=0.7,          # optimal
    ttl_seconds=14 * 24 * 3600,  # 14 ngày
)

# Adaptation config
adaptation = TripletAdaptation(
    sgd_steps=1,
    learning_rate=1e-4,
    weight_decay=1e-6,
    margin=float("inf"),
)

# Engine — M3 gọi 3 hàm này trong endpoints
engine = PersonalizationEngine(
    lifecycle=lifecycle,
    clip_embeddings=clip_embeddings,
    adaptation=adaptation,
    trigger_every_n=9,
    candidate_k=100,
    final_k=10,
)
```

### Bước 3 — Gọi API M2 trong 3 endpoints

#### `/recommend` — Lấy gợi ý cho user

```python
@app.post("/recommend")
def recommend(user_id: str, seed_article_id: str, k: int = 10):
    recs = engine.get_recommendations(user_id, seed_article_id, k)
    return {"recommendations": recs}
```

**Response:**

```json
{
  "recommendations": ["108775044", "108775015", "110065001", ...]
}
```

#### `/interact` — Xử lý tương tác user

```python
@app.post("/interact")
def interact(user_id: str, article_id: str,
             interaction_type: str = "purchase",
             shown_articles: list = []):
    result = engine.handle_interaction(
        user_id=user_id,
        article_id=article_id,
        interaction_type=interaction_type,  # "purchase" | "click" | "view"
        shown_articles=shown_articles,
    )
    return result
```

**Response:**

```json
{
  "status": "ok",
  "interaction_count": 9,
  "adapted": true,
  "adaptation_time_ms": 12.4
}
```

#### `/user-state` — Kiểm tra trạng thái user (optional)

```python
@app.get("/user-state")
def user_state(user_id: str):
    return engine.get_user_state(user_id)
```

**Response:**

```json
{
  "user_id": "user_123",
  "ema_vector": [0.12, -0.34, ...],  // 64 floats
  "interaction_count": 9,
  "has_personalization": true
}
```

---

## 4. Luồng hoạt động end-to-end

```
User mở app
    │
    ├─► GET /recommend?user_id=X&seed=article_Y
    │       └─ Cold user  → item-to-item KNN (CLIP embeddings)
    │       └─ Warm user  → Stage1: recall 100 → Stage2: rerank top-10
    │
    ├─► POST /interact  (user click/purchase)
    │       └─ Update EMA vector
    │       └─ Mỗi 9 interactions → Triplet adaptation
    │       └─ flush Personal MLP → Redis (TTL 14 ngày)
    │
    └─► GET /recommend  (lần tiếp theo — đã personalized)
```

---

## 5. Lưu ý 

- **Key format Redis:** `mlp:{user_id}` — M3 không cần biết nội dung, chỉ cần store/retrieve string base64
- **Personal MLP size:** ~700 KB/user sau serialize — tính toán quota Redis phù hợp
- **LRU cache:** 200 users hot trong memory mỗi container — nếu scale nhiều containers thì mỗi container có LRU riêng, Redis là source of truth
- **Cold start:** User chưa có tương tác → `engine.get_recommendations()` tự fallback sang item-to-item KNN, không cần xử lý riêng
- **Interaction weights:** `purchase=4`, `click=1`, `view=1` — đã hardcode trong `user_state.py`
