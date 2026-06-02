import numpy as np
from sklearn.metrics import roc_auc_score


def _as_bool_matrix(pos_index):
    pos_index = np.asarray(pos_index, dtype=bool)
    if pos_index.ndim == 1:
        pos_index = pos_index.reshape(1, -1)
    return pos_index


def _as_pos_len(pos_len):
    pos_len = np.asarray(pos_len, dtype=float)
    if pos_len.ndim == 0:
        pos_len = pos_len.reshape(1)
    return pos_len


def recall_(pos_index, pos_len):
    pos_index = _as_bool_matrix(pos_index)
    pos_len = _as_pos_len(pos_len)
    rec_ret = np.cumsum(pos_index, axis=1) / pos_len.reshape(-1, 1)
    return rec_ret.mean(axis=0)


def recall2_(pos_index, pos_len):
    pos_index = _as_bool_matrix(pos_index)
    pos_len = _as_pos_len(pos_len)
    rec_cum = np.cumsum(pos_index, axis=1)
    return rec_cum.sum(axis=0) / pos_len.sum()


def precision_(pos_index, pos_len=None):
    pos_index = _as_bool_matrix(pos_index)
    rec_ret = pos_index.cumsum(axis=1) / np.arange(1, pos_index.shape[1] + 1)
    return rec_ret.mean(axis=0)


def ndcg_(pos_index, pos_len):
    pos_index = _as_bool_matrix(pos_index)
    pos_len = _as_pos_len(pos_len).astype(int)
    len_rank = np.full_like(pos_len, pos_index.shape[1])
    idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)

    ranks = np.zeros_like(pos_index, dtype=float)
    ranks[:, :] = np.arange(1, pos_index.shape[1] + 1)

    idcg = np.cumsum(1.0 / np.log2(ranks + 1), axis=1)
    for row, idx in enumerate(idcg_len):
        if idx <= 0:
            idcg[row, :] = np.inf
        else:
            idcg[row, idx:] = idcg[row, idx - 1]

    dcg = 1.0 / np.log2(ranks + 1)
    dcg = np.cumsum(np.where(pos_index, dcg, 0), axis=1)
    return (dcg / idcg).mean(axis=0)


def map_(pos_index, pos_len):
    pos_index = _as_bool_matrix(pos_index)
    pos_len = _as_pos_len(pos_len).astype(int)
    pre = pos_index.cumsum(axis=1) / np.arange(1, pos_index.shape[1] + 1)
    sum_pre = np.cumsum(pre * pos_index.astype(float), axis=1)
    len_rank = np.full_like(pos_len, pos_index.shape[1])
    actual_len = np.where(pos_len > len_rank, len_rank, pos_len)

    result = np.zeros_like(pos_index, dtype=float)
    for row, lens in enumerate(actual_len):
        if lens <= 0:
            continue
        ranges = np.arange(1, pos_index.shape[1] + 1)
        ranges[lens:] = ranges[lens - 1]
        result[row] = sum_pre[row] / ranges
    return result.mean(axis=0)


metrics_dict = {
    'recall': recall_,
    'recall2': recall2_,
    'precision': precision_,
    'ndcg': ndcg_,
    'map': map_,
}


def recall(rank, ground_truth, N):
    return len(set(rank[:N]) & set(ground_truth)) / float(len(set(ground_truth)))


def precision_at_k(r, k):
    return precision_(np.asarray(r)[:k])[-1]


def average_precision(r, cut):
    return map_(np.asarray(r)[:cut], min(cut, np.sum(r)))[-1]


def mean_average_precision(rs):
    return np.mean([average_precision(r, len(r)) for r in rs])


def dcg_at_k(r, k, method=1):
    r = np.asarray(r, dtype=float)[:k]
    if not r.size:
        return 0.
    if method == 0:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    if method == 1:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    raise ValueError('method must be 0 or 1.')


def ndcg_at_k(r, k, all_pos_num=None):
    if all_pos_num is None:
        all_pos_num = np.sum(r)
    if all_pos_num == 0:
        return 0.
    return ndcg_(np.asarray(r)[:k], all_pos_num)[-1]


def recall_at_k(r, k, all_pos_num):
    if all_pos_num == 0:
        return 0.
    return recall_(np.asarray(r)[:k], all_pos_num)[-1]


def map_at_k(r, k, all_pos_num):
    if all_pos_num == 0:
        return 0.
    return map_(np.asarray(r)[:k], all_pos_num)[-1]


def hit_at_k(r, k):
    r = np.asarray(r)[:k]
    return 1. if np.sum(r) > 0 else 0.


def F1(pre, rec):
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    return 0.


def auc(ground_truth, prediction):
    try:
        return roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        return 0.
