# Evaluation

Đánh giá hệ thống fashion recommendation theo paper protocol (Section 3 + Appendix C).

## Yêu cầu

Trước khi chạy, cần có đủ các file sau:

```
data/
  subset_1week/transactions_train.csv   # train week 2020-08-09 → 2020-08-15
  transactions_train_full.csv           # full 2-year dataset (lấy test week)
checkpoints/
  clip_embeddings.pt
  mlp_student.pt
  article_embeddings_hgnn3.pt
```

## Chạy đánh giá

```bash
# Chạy toàn bộ (9195 users, ~4p)
python -m evaluation.run_eval_hm --no-synthetic

# Chạy nhanh với subset users
python -m evaluation.run_eval_hm --no-synthetic --sample 500

# Bỏ qua LightGCN 
python -m evaluation.run_eval_hm --no-synthetic --no-lightgcn

# Tùy chỉnh K và output
python -m evaluation.run_eval_hm --no-synthetic --k 10 --output results/my_results.csv
```

## Tham số

| Tham số           | Mặc định                     | Mô tả                             |
| ------------------ | ------------------------------- | ----------------------------------- |
| `--k`            | 20                              | Số recommendations trả về        |
| `--T`            | 12                              | Số items ground truth              |
| `--sample`       | None                            | Lấy mẫu N users (None = tất cả) |
| `--output`       | `results/eval_hm_results.csv` | Đường dẫn lưu kết quả        |
| `--no-lightgcn`  | False                           | Bỏ qua LightGCN                    |
| `--no-synthetic` | False                           | Không dùng synthetic interactions |

## Kết quả

Kết quả in ra theo định dạng paper Table 2/3, metrics nhân ×10⁴:

```
==========================================================================================
  EVALUATION RESULTS  (K=10, T=12)  —  metrics ×10^4  (paper Table 2/3)
==========================================================================================
Model                                 HR        NDCG   Precision      Recall    F1-score
------------------------------------------------------------------------------------------
  Random Baseline                 15±390        3±85        2±39       5±170        2±54
  Popular Items                 360±1863      95±588      38±198     127±844      52±278
  LightGCN                      867±2814    384±1495      96±320    426±1682     143±487
  HGNN (No Personalization)     465±2107     180±986      50±231    220±1191      75±352
  HGNN + EMA (Ours)             464±2104     179±980      49±230    220±1187      75±351
==========================================================================================
```

File CSV được lưu tại `results/eval_hm_results.csv` (aggregated) và `results/eval_hm_results_per_user.csv` (per-user).

## Protocol

- **Train week** (08-09 → 08-15): fit baselines + build observed session (EMA input)
- **Test week** (08-16 → 08-22): ground truth purchases
- Chỉ evaluate users xuất hiện ở **cả 2 tuần**
- Tất cả baselines dùng **EMA+KNN** (paper Appendix C) để fair comparison

## Cấu trúc file

| File               | Mô tả                                                        |
| ------------------ | -------------------------------------------------------------- |
| `data_split.py`  | Load và split dữ liệu theo paper protocol                   |
| `baselines.py`   | RandomBaseline, PopularItems, LightGCN, HGNNNoEMA, HGNNWithEMA |
| `run_eval_hm.py` | Script chính — chạy tất cả models và in kết quả        |
