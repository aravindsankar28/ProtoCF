import torch
import numpy as np
import metrics
from scipy import sparse
from collections import defaultdict
import torch.nn.functional as F


def evaluate_popular(model, eval_loader, n_users, device, ui_sp_mat=None):
    model.eval()
    eval_loss = 0.0
    if model.base_model == 'vae' or model.base_model == 'dae':
        all_user_embed, _, _ = model.user_encoder(torch.Tensor(ui_sp_mat[torch.arange(n_users).numpy(), :].toarray()).
                                                  to(device).squeeze(), torch.arange(n_users).to(device))
    else:
        all_user_embed = model.user_lookup(torch.arange(n_users).unsqueeze(1).to(device)).squeeze()  # shape [I, D]

    if model.mode != "base":
        all_user_embed_base = model.user_lookup_base(
            torch.arange(n_users).unsqueeze(1).to(device)).squeeze()  # shape [I, D]
    n100_list = []
    r20_list = []
    r50_list = []
    preds = defaultdict(list)
    # compute all item embeddings.
    with torch.no_grad():
        pred_list = []
        pred_list1 = []
        query_list = []
        for data in eval_loader:
            all_support_users, query_users, fold_in, idx, mask = data
            all_support_users = all_support_users.to(device, non_blocking=True)
            query_users = query_users.to(device, non_blocking=True)
            idx = idx.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            pred_scores1 = None
            fold_in = np.array(fold_in)  # binary vector over users indicating training ratings.
            support_users = all_support_users
            if model.mode == 'base':
                item_embed = model.item_lookup(idx)
            else:
                if model.base_model == 'vae' or model.base_model == 'dae':
                    user_embedding, _, _ = model.user_encoder(torch.Tensor(ui_sp_mat[support_users.contiguous().
                                                                           view(-1).cpu().numpy(),
                                                                           :].toarray()).squeeze()
                                                              .to(device, non_blocking=True),
                                                              support_users.contiguous().
                                                              view(-1))
                    user_embedding = user_embedding.view(-1, support_users.shape[1], model.embed_dim_item)
                else:
                    user_embedding = model.user_lookup(support_users)
                item_embed = (user_embedding * mask.unsqueeze(-1)).sum(-2).squeeze() / mask.sum(-1).squeeze().unsqueeze(
                    -1)

                item_embed1 = model.item_lookup(idx)
                if model.mode == 'few-shot' and model.model_variant == 'memory':
                    item_embed, _ = model.memory_embedding(item_embed)

            batch_size = item_embed.shape[0]

            if model.mode != 'base':
                pred_scores1 = (model.cos(all_user_embed_base.unsqueeze(0).repeat(batch_size, 1, 1),
                                          item_embed1.unsqueeze(1).repeat(1, n_users, 1)) * 10).cpu()
                pred_scores = model.cos(all_user_embed.unsqueeze(0).repeat(batch_size, 1, 1),
                                        item_embed.unsqueeze(1).repeat(1, n_users, 1)) * 10
            else:
                pred_scores = model.cos(all_user_embed.unsqueeze(0).repeat(batch_size, 1, 1),
                                        item_embed.unsqueeze(1).repeat(1, n_users, 1)) * 10
            pred_scores[fold_in.nonzero()] = -np.inf
            if model.mode != 'base':
                pred_scores1[fold_in.nonzero()] = -np.inf
                pred_list1.append(pred_scores1)
            pred_list.append(pred_scores.cpu())

            query_list.append(query_users.cpu())
            preds[0].extend(pred_scores.cpu().numpy())

        query_users = torch.cat(query_list).permute(1, 0).contiguous()
        pred_scores = torch.cat(pred_list).permute(1, 0).contiguous()
        if model.mode != 'base':
            pred_scores1 = torch.cat(pred_list1).permute(1, 0).contiguous()

    unique_users = torch.arange(n_users)[~torch.all(query_users == 0, dim=1)].numpy()
    users_split = np.array_split(unique_users, 10)

    for users in users_split:
        if model.mode != 'base':
            curr_pred_scores = pred_scores1[users]
        else:
            curr_pred_scores = pred_scores[users]
        curr_query = query_users[users]

        n_100 = torch.mean(metrics.NDCG_binary_at_k_batch_torch(curr_pred_scores, curr_query, 100))
        r_20 = torch.mean(metrics.Recall_at_k_batch_torch(curr_pred_scores, curr_query, 20))
        r_50 = torch.mean(metrics.Recall_at_k_batch_torch(curr_pred_scores, curr_query, 50))

        n100_list.append(n_100)
        r20_list.append(r_20)
        r50_list.append(r_50)

    n_100 = torch.mean(torch.tensor(n100_list))
    r_20 = torch.mean(torch.tensor(r20_list))
    r_50 = torch.mean(torch.tensor(r50_list))

    num_batches = max(1, len(eval_loader.dataset) / eval_loader.batch_size)
    eval_loss /= num_batches
    return eval_loss, n_100, r_20, r_50, query_users, pred_scores, pred_scores1


def evaluate_few_shot(model, eval_loader, device, query_users, ui_sp_mat=None, sample_size=None, test=False):
    model.eval()
    eval_loss = 0.0
    user_index = torch.LongTensor(query_users[0])
    fewshot_item_emb = []
    if model.base_model == 'vae' or model.base_model == 'dae':
        all_user_embed, _, _ = model.user_encoder(
            torch.Tensor(ui_sp_mat[torch.arange(model.n_users).numpy(), :].toarray()).to(device).squeeze(),
            torch.arange(model.n_users).to(device, non_blocking=True))
    else:
        all_user_embed = model.user_lookup(torch.arange(model.n_users).unsqueeze(1).to(device)).squeeze()
    if model.mode != "base":
        all_user_embed_base = model.user_lookup_base(
            torch.arange(model.n_users).unsqueeze(1).to(device)).squeeze()  # shape [I, D]

    preds = defaultdict(list)
    with torch.no_grad():
        pred_list = []
        pred_list1 = []
        query_list = []
        all_query_list = []
        all_pred_list = []
        for data in eval_loader:
            all_support_users, all_query_users, mask, idx = data
            idx = idx.to(device, non_blocking=True)
            all_support_users = all_support_users.to(device, non_blocking=True)
            all_query_users = all_query_users.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            all_query_list.append(all_query_users[:, 0, :])
            pred_scores1 = None
            support_users = all_support_users[:, 0, :]
            query_users = all_query_users[:, 0, user_index]
            if model.base_model == 'vae' or model.base_model == 'dae':
                user_embedding, _, _ = model.user_encoder(torch.Tensor(ui_sp_mat[support_users.contiguous().
                                                                       view(-1).cpu().numpy(), :].toarray()).squeeze()
                                                          .to(device, non_blocking=True), support_users.contiguous().
                                                          view(-1))
                user_embedding = user_embedding.view(-1, support_users.shape[1], model.embed_dim_item)
            else:
                user_embedding = model.user_lookup(support_users)
            item_embed = model.prototype(user_embedding, mask=mask)
            item_embed1 = model.item_lookup(idx + eval_loader.dataset.start_idx)
            if model.mode == 'few-shot' and model.model_variant == 'memory':
                item_embed, _ = model.memory_embedding(item_embed)
            if test:
                fewshot_item_emb.append(item_embed.detach().cpu())
            batch_size = item_embed.shape[0]
            item_vec = item_embed.unsqueeze(1).repeat(1, all_user_embed.shape[0], 1)
            user_vec = all_user_embed.unsqueeze(0).repeat(batch_size, 1, 1)

            pred_scores = model.cos(user_vec, item_vec) * 10
            pred_scores1 = model.cos(all_user_embed_base.unsqueeze(0).repeat(batch_size, 1, 1),
                                     item_embed1.unsqueeze(1).repeat(1, all_user_embed.shape[0], 1)) * 10
            idx = torch.arange(batch_size).unsqueeze(1).repeat(1, support_users.shape[1]).view(-1).numpy()

            temp_users = support_users.view(-1).cpu().numpy()

            temp_support = sparse.csr_matrix((np.ones_like(idx),
                                              (idx, temp_users)), dtype='float32',
                                             shape=(batch_size, model.n_users))

            pred_scores[temp_support.nonzero()] = -np.inf
            pred_scores1[temp_support.nonzero()] = -np.inf
            all_pred_list.append(pred_scores)
            pred_scores = pred_scores[:, user_index]
            pred_list.append(pred_scores)
            pred_list1.append(pred_scores1)
            query_list.append(query_users)
            preds[0].extend(pred_scores.cpu().numpy())
        query_users = torch.cat(query_list).permute(1, 0)
        pred_scores = torch.cat(pred_list).permute(1, 0)
        pred_scores1 = torch.cat(pred_list1).permute(1, 0)
        all_query = torch.cat(all_query_list).permute(1, 0)
        all_pred = torch.cat(all_pred_list).permute(1, 0)
        n_100 = torch.mean(metrics.NDCG_binary_at_k_batch_torch(pred_scores, query_users, 100, device=device))
        r_20 = torch.mean(metrics.Recall_at_k_batch_torch(pred_scores, query_users, 20))
        r_50 = torch.mean(metrics.Recall_at_k_batch_torch(pred_scores, query_users, 50))

    num_batches = max(1, len(eval_loader.dataset) / eval_loader.batch_size)
    eval_loss /= num_batches
    return eval_loss, n_100, r_20, r_50, all_query, all_pred, pred_scores1


def evaluate_joint(model, query_tail, query_head, tail_pred, head_pred, head_pred1, tail_pred1=None, lambd=0):
    tail_pred = tail_pred.cpu()
    tail_pred1 = tail_pred1.cpu()
    head_pred = head_pred.cpu()
    head_pred1 = head_pred1.cpu()
    query_head = query_head.cpu()
    query_tail = query_tail.cpu()
    assert tail_pred.shape == query_tail.shape
    assert head_pred.shape == query_head.shape
    assert head_pred.shape == head_pred1.shape
    model.eval()
    print("Current Lambda: ", lambd)
    n20_list = []
    n50_list = []
    r20_list = []
    r50_list = []
    tail_pred = (1 - lambd) * tail_pred + lambd * tail_pred1
    head_pred = (1 - lambd) * head_pred + lambd * head_pred1

    all_pred = torch.cat([head_pred, tail_pred], dim=1).contiguous().cpu().numpy()
    query_users = torch.cat([query_head, query_tail], dim=1).contiguous().cpu()

    unique_users = torch.arange(model.n_users)[~torch.all(query_users == 0, dim=1)].numpy()
    head_users = torch.arange(model.n_users)[~torch.all(query_head == 0, dim=1)].numpy()
    tail_users = torch.arange(model.n_users)[~torch.all(query_tail == 0, dim=1)].numpy()
    n_20 = torch.mean(
        metrics.NDCG_binary_at_k_batch_torch(tail_pred[tail_users], query_tail[tail_users], 20, device='cpu'))
    n_50 = torch.mean(
        metrics.NDCG_binary_at_k_batch_torch(tail_pred[tail_users], query_tail[tail_users], 50, device='cpu'))
    r_20 = torch.mean(metrics.Recall_at_k_batch_torch(tail_pred[tail_users], query_tail[tail_users], 20))
    r_50 = torch.mean(metrics.Recall_at_k_batch_torch(tail_pred[tail_users], query_tail[tail_users], 50))

    print('| Few-shot test evaluation | n20 {:4.4f} | n50 {:4.4f} | r20 {:4.4f} | '
          'r50 {:4.4f}'.format(n_20, n_50, r_20, r_50))

    n_20 = torch.mean(
        metrics.NDCG_binary_at_k_batch_torch(head_pred[head_users], query_head[head_users], 20, device='cpu'))
    n_50 = torch.mean(
        metrics.NDCG_binary_at_k_batch_torch(head_pred[head_users], query_head[head_users], 50, device='cpu'))
    r_20 = torch.mean(metrics.Recall_at_k_batch_torch(head_pred[head_users], query_head[head_users], 20))
    r_50 = torch.mean(metrics.Recall_at_k_batch_torch(head_pred[head_users], query_head[head_users], 50))
    print('| Popular test evaluation | n20 {:4.4f} | n50 {:4.4f} | r20 {:4.4f} | '
          'r50 {:4.4f}'.format(n_20, n_50, r_20, r_50))

    users_split = np.array_split(unique_users, 10)

    for users in users_split:
        curr_pred_scores = torch.cat([head_pred[users], tail_pred[users]], dim=1).contiguous()
        curr_query = query_users[users]

        n_20 = torch.mean(metrics.NDCG_binary_at_k_batch_torch(curr_pred_scores, curr_query, 20, 'cpu'))
        n_50 = torch.mean(metrics.NDCG_binary_at_k_batch_torch(curr_pred_scores, curr_query, 50, 'cpu'))
        r_20 = torch.mean(metrics.Recall_at_k_batch_torch(curr_pred_scores, curr_query, 20))
        r_50 = torch.mean(metrics.Recall_at_k_batch_torch(curr_pred_scores, curr_query, 50))

        n20_list.append(n_20)
        n50_list.append(n_50)
        r20_list.append(r_20)
        r50_list.append(r_50)
    n_20 = torch.mean(torch.tensor(n20_list))
    n_50 = torch.mean(torch.tensor(n50_list))
    r_20 = torch.mean(torch.tensor(r20_list))
    r_50 = torch.mean(torch.tensor(r50_list))

    return n_20, n_50, r_20, r_50, all_pred
