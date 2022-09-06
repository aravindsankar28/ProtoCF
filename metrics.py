import bottleneck as bn
import numpy as np
import torch


def NDCG_binary_at_k_batch_torch(X_pred, heldout_batch, k=100, device='cpu'):
    """
    Normalized Discounted Cumulative Gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    """
    batch_users = X_pred.shape[0]  # batch_size
    _, idx_topk = torch.topk(X_pred, k, dim=1, sorted=True)
    tp = 1.0 / torch.log2(torch.arange(2, k + 2, device=device).float())
    heldout_batch_nonzero = (heldout_batch > 0).float()
    DCG = (heldout_batch_nonzero[torch.arange(batch_users, device=device).unsqueeze(1), idx_topk] * tp).sum(dim=1)
    heldout_nonzero = (heldout_batch > 0).sum(dim=1)  # num. of non-zero items per batch. [B]
    IDCG = torch.FloatTensor([(tp[:min(n, k)]).sum() for n in heldout_nonzero]).to(device)
    return DCG / IDCG


def Recall_at_k_batch(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]
    idx = bn.argpartition(-X_pred, k, axis=1)  # indices of the top-k items (not in order).
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)  # [B, I]
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True
    X_true_binary = (heldout_batch > 0)  # .toarray() #  [B, I]
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall


def Recall_at_k_batch_torch(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]
    _, topk_indices = torch.topk(X_pred, k, dim=1, sorted=False)  # [B, K]
    X_pred_binary = torch.zeros_like(X_pred)
    if torch.cuda.is_available():
        X_pred_binary = X_pred_binary.cuda()
    X_pred_binary[torch.arange(batch_users).unsqueeze(1), topk_indices] = 1
    X_true_binary = (heldout_batch > 0).float()  # .toarray() #  [B, I]
    k_tensor = torch.Tensor([k])
    if torch.cuda.is_available():
        X_true_binary = X_true_binary.cuda()
        k_tensor = k_tensor.cuda()
    tmp = (X_true_binary * X_pred_binary).sum(dim=1).float()
    recall = tmp / torch.min(k_tensor, X_true_binary.sum(dim=1).float())
    return recall


def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=100):
    """
    Normalized Discounted Cumulative Gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    """
    batch_users = X_pred.shape[0]  # batch_size
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)

    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk] * tp).sum(axis=1)

    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in np.count_nonzero(heldout_batch, axis=1)])
    return DCG / IDCG
