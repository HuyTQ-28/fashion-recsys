"""
Baselines cho đánh giá hệ thống fashion recommendation.

Tất cả baselines tuân theo unified EMA+KNN evaluation protocol (paper Appendix C):
  1. Train model → học item embeddings
  2. User vector = EMA của item embeddings theo thứ tự thời gian (paper eq.5)
  3. Recommend = K-NN search từ EMA vector trong item embedding space
  4. Đánh giá: Precision@K, Recall@K, F1@K (×10^4), K=10, T=12
"""

from __future__ import annotations

import copy
import logging
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


class RandomBaseline:
    """Recommend K random items từ catalog — lower bound."""

    def __init__(self, seed: int = 42):
        self._catalog: List[str] = []
        self._rng = np.random.default_rng(seed)

    def fit(self, train_purchases: pd.DataFrame, catalog: Optional[List[str]] = None) -> "RandomBaseline":
        if catalog is not None:
            self._catalog = list(catalog)
        else:
            self._catalog = train_purchases["article_id"].astype(str).unique().tolist()
        logger.info(f"RandomBaseline: catalog size {len(self._catalog)}")
        return self

    def recommend(self, user_id: str, k: int = 10) -> List[str]:
        if not self._catalog:
            return []
        chosen = self._rng.choice(self._catalog, size=min(k, len(self._catalog)), replace=False)
        return chosen.tolist()


class LastKBaseline:
    """Recommend K items cuối cùng user đã mua trong train window (paper Table 2)."""

    def __init__(self):
        self._user_last_items: Dict[str, List[str]] = {}

    def fit(self, train_purchases: pd.DataFrame) -> "LastKBaseline":
        df = train_purchases[["customer_id", "article_id"]].copy()
        if "t_dat" in train_purchases.columns:
            df["t_dat"] = train_purchases["t_dat"]
            df = df.sort_values("t_dat")
        df["customer_id"] = df["customer_id"].astype(str)
        df["article_id"]  = df["article_id"].astype(str)

        for user_id, group in df.groupby("customer_id"):
            seen = {}
            for aid in group["article_id"]:
                seen[aid] = True
            self._user_last_items[user_id] = list(seen.keys())

        logger.info(f"LastKBaseline: built history for {len(self._user_last_items)} users")
        return self

    def recommend(self, user_id: str, k: int = 10) -> List[str]:
        items = self._user_last_items.get(user_id, [])
        return list(reversed(items))[:k]


class PopularItems:
    """Gợi ý top-K sản phẩm được mua nhiều nhất trong train window."""

    def __init__(self):
        self._top_items: List[str] = []

    def fit(self, train_purchases: pd.DataFrame) -> "PopularItems":
        counts = Counter(train_purchases["article_id"].astype(str))
        self._top_items = [item for item, _ in counts.most_common()]
        logger.info(f"PopularItems: tracked {len(self._top_items)} items")
        return self

    def recommend(self, user_id: str, k: int = 20) -> List[str]:
        return self._top_items[:k]


class LightGCNBaseline:
    """
    LightGCN adapted theo paper Appendix C:
      1. Build item-item co-purchase graph: edge(i,j) weight = số users mua cả i lẫn j
      2. Train GNN với BPR loss → 64-dim item embeddings
      3. User vector = EMA của item embeddings (eq.5)
      4. Recommend = KNN từ EMA user vector
    """

    def __init__(
        self,
        factors: int = 64,
        n_layers: int = 2,
        iterations: int = 20,
        lr: float = 0.01,
        regularization: float = 1e-5,
        ema_alpha: float = 0.7,
    ):
        self.factors = factors
        self.n_layers = n_layers
        self.iterations = iterations
        self.lr = lr
        self.regularization = regularization
        self.ema_alpha = ema_alpha

        self._item_emb: Optional[torch.Tensor] = None
        self._item_index: Dict[str, int] = {}
        self._items: List[str] = []
        self._user_ema: Dict[str, torch.Tensor] = {}

    def fit(self, train_purchases: pd.DataFrame) -> "LightGCNBaseline":
        df = train_purchases[["customer_id", "article_id"]].copy()
        df["customer_id"] = df["customer_id"].astype(str)
        df["article_id"]  = df["article_id"].astype(str)

        items = df["article_id"].unique().tolist()
        self._item_index = {it: i for i, it in enumerate(items)}
        self._items = items
        n_items = len(items)

        user_items: Dict[str, List[int]] = {}
        for _, row in df.iterrows():
            uid = row["customer_id"]
            iid = self._item_index[row["article_id"]]
            user_items.setdefault(uid, []).append(iid)

        from collections import defaultdict
        edge_weight: Dict[tuple, int] = defaultdict(int)
        for uid, iids in user_items.items():
            unique_iids = list(set(iids))
            for a in range(len(unique_iids)):
                for b in range(a + 1, len(unique_iids)):
                    i, j = unique_iids[a], unique_iids[b]
                    edge_weight[(min(i,j), max(i,j))] += 1

        if not edge_weight:
            logger.warning("LightGCN: no co-purchase edges found")
            self._item_emb = torch.zeros(n_items, self.factors)
            return self

        pairs = list(edge_weight.keys())
        weights = [float(edge_weight[p]) for p in pairs]
        row_idx = torch.tensor([p[0] for p in pairs] + [p[1] for p in pairs], dtype=torch.long)
        col_idx = torch.tensor([p[1] for p in pairs] + [p[0] for p in pairs], dtype=torch.long)
        edge_vals = torch.tensor(weights + weights, dtype=torch.float)

        # D^{-1/2} A D^{-1/2} normalisation — tránh bias về item phổ biến
        deg = torch.zeros(n_items)
        deg.scatter_add_(0, row_idx, edge_vals)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
        norm_vals = deg_inv_sqrt[row_idx] * edge_vals * deg_inv_sqrt[col_idx]

        adj = torch.sparse_coo_tensor(
            torch.stack([row_idx, col_idx]), norm_vals, (n_items, n_items)
        ).coalesce()

        logger.info(f"LightGCN: {n_items} items, {len(pairs)} co-purchase edges, "
                    f"{self.n_layers} layers, {self.iterations} epochs...")

        E = torch.nn.Parameter(torch.randn(n_items, self.factors) * 0.01)
        optimizer = torch.optim.Adam([E], lr=self.lr)

        pos_arr = np.array(pairs, dtype=np.int64)
        batch_size = min(4096, len(pos_arr))

        def gnn_forward() -> torch.Tensor:
            all_embs = [E]
            emb = E
            for _ in range(self.n_layers):
                emb = torch.sparse.mm(adj, emb)
                all_embs.append(emb)
            return torch.stack(all_embs, dim=0).mean(dim=0)

        for epoch in range(self.iterations):
            idx = np.random.permutation(len(pos_arr))
            total_loss = 0.0
            n_batches = 0
            for start in range(0, len(idx), batch_size):
                batch = pos_arr[idx[start: start + batch_size]]
                i_a   = torch.from_numpy(batch[:, 0]).long()
                i_pos = torch.from_numpy(batch[:, 1]).long()
                # In-batch negatives: tránh false negatives khi sample ngẫu nhiên toàn catalog
                i_neg = i_pos[torch.randperm(len(i_pos))]

                ea  = E[i_a]
                epo = E[i_pos]
                ene = E[i_neg]

                pos_score = (ea * epo).sum(dim=1)
                neg_score = (ea * ene).sum(dim=1)
                bpr_loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()
                reg_loss = self.regularization * (
                    ea.norm(2).pow(2) + epo.norm(2).pow(2) + ene.norm(2).pow(2)
                )

                optimizer.zero_grad()
                (bpr_loss + reg_loss).backward()
                optimizer.step()
                total_loss += bpr_loss.item()
                n_batches += 1

            if (epoch + 1) % 5 == 0:
                logger.info(f"  Epoch {epoch+1}/{self.iterations} loss={total_loss/n_batches:.4f}")

        with torch.no_grad():
            self._item_emb = gnn_forward().detach()

        logger.info("LightGCN: training complete")
        return self

    def build_user_ema_vectors(self, train_purchases: pd.DataFrame) -> "LightGCNBaseline":
        df = train_purchases[["customer_id", "article_id"]].copy()
        if "t_dat" in train_purchases.columns:
            df["t_dat"] = train_purchases["t_dat"]
            df = df.sort_values("t_dat")
        df["customer_id"] = df["customer_id"].astype(str)
        df["article_id"]  = df["article_id"].astype(str)

        alpha = self.ema_alpha
        for user_id, group in df.groupby("customer_id"):
            ema = None
            for aid in group["article_id"]:
                if aid not in self._item_index:
                    continue
                item_vec = self._item_emb[self._item_index[aid]]
                ema = item_vec if ema is None else (1 - alpha) * ema + alpha * item_vec
            if ema is not None:
                self._user_ema[user_id] = ema

        logger.info(f"LightGCN: built EMA vectors for {len(self._user_ema)} users")
        return self

    @torch.no_grad()
    def recommend(self, user_id: str, k: int = 10) -> List[str]:
        if self._item_emb is None or user_id not in self._user_ema:
            return []
        u_vec = self._user_ema[user_id].unsqueeze(0)
        distances = torch.cdist(u_vec, self._item_emb).squeeze(0)
        _, indices = distances.topk(k, largest=False)
        return [self._items[i] for i in indices.tolist()]


class HGNNNoEMA:
    """
    HGNN embeddings + Student MLP không fine-tune — ablation (no personalization).
    User vector = EMA tĩnh từ train history, không cập nhật theo context mới.
    """

    def __init__(
        self,
        clip_checkpoint: str = "checkpoints/clip_embeddings.pt",
        mlp_checkpoint: str = "checkpoints/mlp_student.pt",
        hgnn_checkpoint: str = "checkpoints/article_embeddings_hgnn3.pt",
    ):
        self.clip_checkpoint = clip_checkpoint
        self.mlp_checkpoint = mlp_checkpoint
        self.hgnn_checkpoint = hgnn_checkpoint

        self._clip_embs: Dict[str, torch.Tensor] = {}
        self._hgnn_embs: Dict[str, torch.Tensor] = {}
        self._mlp: Optional[torch.nn.Module] = None
        self._user_vectors: Dict[str, torch.Tensor] = {}
        self._item_ids_cache: List[str] = []
        self._item_matrix_cache: Optional[torch.Tensor] = None

    def load(self) -> "HGNNNoEMA":
        self._clip_embs = torch.load(self.clip_checkpoint, map_location="cpu")
        logger.info(f"HGNNNoEMA: loaded {len(self._clip_embs)} CLIP embeddings")

        hgnn_data = torch.load(self.hgnn_checkpoint, map_location="cpu")
        emb_matrix = hgnn_data["embeddings"]
        id2idx: Dict[str, int] = hgnn_data["id2idx"]
        self._hgnn_embs = {
            str(aid): emb_matrix[idx] for aid, idx in id2idx.items()
        }
        logger.info(f"HGNNNoEMA: loaded {len(self._hgnn_embs)} HGNN embeddings")

        from src.models.student_mlp import StudentMLP
        self._mlp = StudentMLP()
        state = torch.load(self.mlp_checkpoint, map_location="cpu")
        # Remap legacy "net.*" keys sang "network.*"
        if any(k.startswith("net.") for k in state):
            state = {k.replace("net.", "network.", 1): v for k, v in state.items()}
        self._mlp.load_state_dict(state)
        self._mlp.eval()
        logger.info("HGNNNoEMA: loaded Student MLP")

        # Pre-compute MLP(CLIP) cho toàn catalog một lần — tái sử dụng trong recommend()
        with torch.no_grad():
            self._item_ids_cache = [aid for aid in self._clip_embs if aid in self._hgnn_embs]
            clip_batch = torch.stack([self._clip_embs[aid] for aid in self._item_ids_cache])
            self._item_matrix_cache = self._mlp(clip_batch)
        return self

    def build_user_vectors(self, train_purchases: pd.DataFrame, ema_alpha: float = 0.7) -> "HGNNNoEMA":
        """EMA của MLP(CLIP) theo thứ tự thời gian (paper eq.5). MLP không fine-tune."""
        df = train_purchases[["customer_id", "article_id"]].copy()
        if "t_dat" in train_purchases.columns:
            df["t_dat"] = train_purchases["t_dat"]
            df = df.sort_values("t_dat")
        df["customer_id"] = df["customer_id"].astype(str)
        df["article_id"]  = df["article_id"].astype(str).str.zfill(10)

        with torch.no_grad():
            for user_id, group in df.groupby("customer_id"):
                ema = None
                for aid in group["article_id"]:
                    if aid not in self._clip_embs:
                        continue
                    proj = self._mlp(self._clip_embs[aid].unsqueeze(0)).squeeze(0)
                    ema = proj if ema is None else (1 - ema_alpha) * ema + ema_alpha * proj
                if ema is not None:
                    self._user_vectors[user_id] = ema

        logger.info(f"HGNNNoEMA: built EMA vectors for {len(self._user_vectors)} users")
        return self

    @torch.no_grad()
    def recommend(self, user_id: str, k: int = 20) -> List[str]:
        if user_id not in self._user_vectors:
            norms = self._item_matrix_cache.norm(dim=1)
            _, indices = norms.topk(k, largest=True)
            return [self._item_ids_cache[i] for i in indices.tolist()]

        user_vec = self._user_vectors[user_id].unsqueeze(0)
        distances = torch.cdist(user_vec, self._item_matrix_cache).squeeze(0)
        _, indices = distances.topk(k, largest=False)
        return [self._item_ids_cache[i] for i in indices.tolist()]


class HGNNWithEMAInProcess:
    """
    Our System — HGNN + Personal MLP + EMA (paper Section 2 + Appendix C).

    Per user:
      1. Personal MLP = deep copy Student MLP, fine-tuned qua Triplet Loss (Eq.6-7)
      2. User EMA = EMA của Personal MLP projections theo thứ tự thời gian (Eq.5)
      3. Recommend = KNN trong Personal MLP space từ EMA vector
    """

    def __init__(
        self,
        clip_checkpoint: str = "checkpoints/clip_embeddings.pt",
        mlp_checkpoint: str = "checkpoints/mlp_student.pt",
        hgnn_checkpoint: str = "checkpoints/article_embeddings_hgnn3.pt",
        ema_alpha: float = 0.7,
        sgd_steps: int = 1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-6,
        margin: float = 1.0,
        neg_per_pos: int = 1,
    ):
        self.clip_checkpoint = clip_checkpoint
        self.mlp_checkpoint = mlp_checkpoint
        self.hgnn_checkpoint = hgnn_checkpoint
        self.ema_alpha = ema_alpha
        self.sgd_steps = sgd_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.margin = margin
        self.neg_per_pos = neg_per_pos

        self._clip_embs: Dict[str, torch.Tensor] = {}
        self._hgnn_embs: Dict[str, torch.Tensor] = {}
        self._base_mlp_state: dict = {}
        self._catalog_ids: List[str] = []
        self._user_ema: Dict[str, torch.Tensor] = {}
        self._user_mlp_state: Dict[str, dict] = {}
        self._item_ids_cache: List[str] = []
        self._clip_batch_cache: Optional[torch.Tensor] = None

    def load(self) -> "HGNNWithEMAInProcess":
        self._clip_embs = torch.load(self.clip_checkpoint, map_location="cpu")
        logger.info(f"HGNNWithEMA: loaded {len(self._clip_embs)} CLIP embeddings")

        hgnn_data = torch.load(self.hgnn_checkpoint, map_location="cpu")
        emb_matrix = hgnn_data["embeddings"]
        id2idx: Dict[str, int] = hgnn_data["id2idx"]
        self._hgnn_embs = {str(aid): emb_matrix[idx] for aid, idx in id2idx.items()}
        logger.info(f"HGNNWithEMA: loaded {len(self._hgnn_embs)} HGNN embeddings")

        from src.models.student_mlp import StudentMLP
        _mlp = StudentMLP()
        state = torch.load(self.mlp_checkpoint, map_location="cpu")
        if any(k.startswith("net.") for k in state):
            state = {k.replace("net.", "network.", 1): v for k, v in state.items()}
        _mlp.load_state_dict(state)
        self._base_mlp_state = {k: v.clone() for k, v in _mlp.state_dict().items()}
        self._catalog_ids = [aid for aid in self._clip_embs if aid in self._hgnn_embs]
        self._item_ids_cache = self._catalog_ids
        # Cache CLIP batch — dùng lại khi project items qua Personal MLP lúc recommend
        self._clip_batch_cache = torch.stack([self._clip_embs[aid] for aid in self._item_ids_cache])
        logger.info("HGNNWithEMA: loaded Student MLP (template for Personal MLP)")
        return self

    def _make_personal_mlp(self) -> torch.nn.Module:
        from src.models.student_mlp import StudentMLP
        mlp = StudentMLP()
        mlp.load_state_dict(copy.deepcopy(self._base_mlp_state))
        return mlp

    def _adapt_mlp(
        self,
        mlp: torch.nn.Module,
        pos_clip: List[torch.Tensor],
        neg_clip: List[torch.Tensor],
        pos_weights: Optional[List[float]] = None,
    ) -> None:
        """Triplet Loss adaptation in-place (paper Eq.6-7)."""
        if not pos_clip or not neg_clip:
            return

        mlp.train()
        optimizer = torch.optim.SGD(
            mlp.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        pos_stack = torch.stack(pos_clip)
        neg_stack = torch.stack(neg_clip)

        # Eq.6: weighted centroid h_wc = (1/|B|) * sum(h_CNN * w)
        if pos_weights is not None:
            w = torch.tensor(pos_weights, dtype=torch.float).unsqueeze(1)
            h_wc = (pos_stack * w).sum(dim=0, keepdim=True) / len(pos_clip)
        else:
            h_wc = pos_stack.mean(dim=0, keepdim=True)

        for _ in range(self.sgd_steps):
            optimizer.zero_grad()
            proj_wc  = mlp(h_wc)
            proj_pos = mlp(pos_stack)
            proj_neg = mlp(neg_stack)

            n_pairs  = min(proj_pos.shape[0], proj_neg.shape[0])
            dist_pos = ((proj_wc - proj_pos[:n_pairs]) ** 2).sum(dim=-1)
            dist_neg = ((proj_wc - proj_neg[:n_pairs]) ** 2).sum(dim=-1)

            # Eq.7: triplet loss với margin
            loss = torch.clamp(dist_pos - dist_neg + self.margin, min=0.0).sum()
            if loss.item() > 0:
                loss.backward()
                optimizer.step()

        mlp.eval()

    def _build_ema_from_mlp(
        self,
        mlp: torch.nn.Module,
        article_ids: List[str],
    ) -> Optional[torch.Tensor]:
        """Rebuild EMA từ Personal MLP đã fine-tune theo thứ tự thời gian (Eq.5)."""
        alpha = self.ema_alpha
        ema = None
        with torch.no_grad():
            for aid in article_ids:
                if aid not in self._clip_embs:
                    continue
                proj = mlp(self._clip_embs[aid].unsqueeze(0)).squeeze(0)
                ema = proj if ema is None else (1 - alpha) * ema + alpha * proj
        return ema

    def build_and_adapt(
        self,
        train_purchases: pd.DataFrame,
        synthetic_interactions: Optional[pd.DataFrame] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> "HGNNWithEMAInProcess":
        """
        Per user: tạo Personal MLP → triplet adaptation → rebuild EMA.
        Tương đương paper "HGNN + 1 week personalization".
        """
        INTERACTION_WEIGHTS = {"purchase": 4, "cart": 3, "favorite": 2, "click": 1}

        if rng is None:
            rng = np.random.default_rng(42)

        df = train_purchases[["customer_id", "article_id"]].copy()
        df["event_type"] = "purchase"
        if "t_dat" in train_purchases.columns:
            df["t_dat"] = train_purchases["t_dat"]

        if synthetic_interactions is not None and len(synthetic_interactions) > 0:
            si = synthetic_interactions[["customer_id", "article_id", "event_type"]].copy()
            if "t_dat" in synthetic_interactions.columns:
                si["t_dat"] = synthetic_interactions["t_dat"]
            else:
                si["t_dat"] = pd.NaT
            df = pd.concat([df, si], ignore_index=True)
            logger.info(f"HGNNWithEMA: merged synthetic interactions ({len(si):,} rows)")

        if "t_dat" in df.columns:
            df = df.sort_values("t_dat", na_position="last")

        df["customer_id"] = df["customer_id"].astype(str)
        df["article_id"]  = df["article_id"].astype(str).str.zfill(10)
        df["event_type"]  = df["event_type"].fillna("purchase")

        catalog_arr = np.array(self._catalog_ids)
        n_adapted = 0

        for user_id, group in df.groupby("customer_id"):
            interactions = [
                (row["article_id"], INTERACTION_WEIGHTS.get(row["event_type"], 1))
                for _, row in group.iterrows()
                if row["article_id"] in self._clip_embs
            ]
            if not interactions:
                continue

            aids    = [aid for aid, _ in interactions]
            weights = [w   for _,   w in interactions]

            personal_mlp = self._make_personal_mlp()
            pos_clip     = [self._clip_embs[aid] for aid in aids]
            pos_weights  = [float(w) for w in weights]

            pos_set    = set(aids)
            neg_size   = max(len(pos_clip) * self.neg_per_pos, 1)
            candidates = [a for a in catalog_arr if a not in pos_set]
            if candidates:
                neg_ids  = rng.choice(candidates, size=min(neg_size, len(candidates)), replace=False)
                neg_clip = [self._clip_embs[nid] for nid in neg_ids if nid in self._clip_embs]
            else:
                neg_clip = []

            self._adapt_mlp(personal_mlp, pos_clip, neg_clip, pos_weights)

            ema = self._build_ema_from_mlp(personal_mlp, aids)
            if ema is not None:
                self._user_ema[user_id] = ema
                self._user_mlp_state[user_id] = {k: v.clone() for k, v in personal_mlp.state_dict().items()}
                n_adapted += 1

        logger.info(f"HGNNWithEMA: adapted Personal MLP + rebuilt EMA for {n_adapted} users")
        return self

    def build_user_ema_vectors(
        self,
        train_purchases: pd.DataFrame,
    ) -> "HGNNWithEMAInProcess":
        return self.build_and_adapt(train_purchases)

    @torch.no_grad()
    def recommend(self, user_id: str, k: int = 20) -> List[str]:
        """KNN trong Personal MLP_u space (paper: 'K-NN to u_t in space projected by MLP_u')."""
        from src.models.student_mlp import StudentMLP
        mlp = StudentMLP()
        if user_id in self._user_mlp_state:
            mlp.load_state_dict(self._user_mlp_state[user_id])
        else:
            mlp.load_state_dict(self._base_mlp_state)
        mlp.eval()

        item_matrix = mlp(self._clip_batch_cache)

        if user_id not in self._user_ema:
            norms = item_matrix.norm(dim=1)
            _, indices = norms.topk(k, largest=True)
            return [self._item_ids_cache[i] for i in indices.tolist()]

        user_vec = self._user_ema[user_id].unsqueeze(0)
        distances = torch.cdist(user_vec, item_matrix).squeeze(0)
        _, indices = distances.topk(k, largest=False)
        return [self._item_ids_cache[i] for i in indices.tolist()]
