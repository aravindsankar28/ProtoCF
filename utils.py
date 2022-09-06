import os
import pandas as pd
from scipy import sparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data
import time
import random


def get_pairs_base(base_item_embed):
    # base_item_embed: B * D
    batch_size = base_item_embed.shape[0]
    x1 = base_item_embed.unsqueeze(1).repeat(1, batch_size, 1)
    x2 = base_item_embed.unsqueeze(0).repeat(batch_size, 1, 1)
    sim_score = F.cosine_similarity(x1, x2, dim=2)  # [B, B]
    positive = torch.LongTensor(torch.topk(sim_score, dim=1, k=5)[1])
    negative = torch.LongTensor(torch.topk(-sim_score, dim=1, k=batch_size - 5)[1])
    return positive, negative


def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return int(np.argmax(memory_available))


def sparse2torch_sparse_l2(data):
    """
    Convert scipy sparse matrix to torch sparse tensor with L2 Normalization
    This is much faster than naive use of torch.FloatTensor(data.toarray())
    https://discuss.pytorch.org/t/sparse-tensor-use-cases/22047/2
    """
    samples = data.shape[0]
    features = data.shape[1]
    coo_data = data.tocoo()
    indices = torch.LongTensor([coo_data.row, coo_data.col])
    row_norms_inv = 1 / np.sqrt(data.sum(1))
    row2val = {i: row_norms_inv[i].item() for i in range(samples)}
    values = np.array([row2val[r] for r in coo_data.row])
    t = torch.sparse.FloatTensor(indices, torch.from_numpy(values).float(), [samples, features])
    return t


def sparse2torch_sparse(data):
    """
    Convert scipy sparse matrix to torch sparse tensor
    This is much faster than naive use of torch.FloatTensor(data.toarray())
    https://discuss.pytorch.org/t/sparse-tensor-use-cases/22047/2
    """
    samples = data.shape[0]
    features = data.shape[1]
    coo_data = data.tocoo()
    indices = torch.LongTensor([coo_data.row, coo_data.col])
    values = data.data
    t = torch.sparse.FloatTensor(indices, torch.from_numpy(values).float(), [samples, features])
    return t


def indices2torch_sparse(data, length):
    """
    Convert a 1d tensor with non-zero indices into a 1d torch sparse tensor
    """
    row = torch.zeros(len(data), dtype=torch.long)
    indices = torch.LongTensor(torch.stack([row, data]))
    val = torch.ones(len(data))
    return torch.sparse.FloatTensor(indices, val, torch.Size([1, length]))


def get_samples(idx, graph, sample_sizes,):
    _, related_items = graph[idx].nonzero()
    one_hop_neighbors = np.random.choice(related_items, sample_sizes[0], replace=True).tolist()
    if len(sample_sizes) < 2:
        return torch.LongTensor(one_hop_neighbors)

    result = []

    for n in one_hop_neighbors:
        _, curr_items = graph[n].nonzero()
        neighbors_n = [n] + np.random.choice(curr_items, sample_sizes[1] - 1, replace=True).tolist()
        result.append(neighbors_n)
    result = torch.LongTensor(np.array(result, dtype=np.int32))

    assert result.shape[0] == sample_sizes[0]
    assert result.shape[1] == sample_sizes[1]
    return result.view(-1)


class TrainDatasetProNet(data.Dataset):
    """
    Load dataset
    """

    def __init__(self, dataset):
        """
        Initialize the PrototypicalDataset object
        Args:
        - dataset: the filename of the dataset used to train, stored in data folder
        - num_samples:  num of samples for each
                           iteration for each class/user (support + query)
        """
        self.dataset = dataset
        self.support_n_samples = 10
        self.query_n_samples = 5

        self.data_dir = os.path.join('../../data/', dataset + "_processed_item")
        self.train_data_ui = self._load_train_data()
        self.train_data_ui_lil = self.train_data_ui.tolil()
        self.item_list = range(0, self.n_items)
        print(f'Loading training data...')

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, idx):
        """
        idx: a tensor of idx of sampled users, single valued tensor.
        @return: (S_k, Q_k) A tuple of sampled items
                 for this sampled user
        """
        users = np.array(self.train_data_ui_lil.rows[idx])
        ttl_size = len(users)  # total number of liked items from this user

        users_tensor = torch.from_numpy(users).long()
        sampled_indices = torch.randperm(ttl_size)  # randomly permute item set.
        if len(users_tensor) > 5:
            query_users = users_tensor[sampled_indices][:self.query_n_samples]
        else:
            query_users = torch.from_numpy(np.random.choice(users, self.query_n_samples, replace=True)).long()
        if len(users_tensor) < 15:
            if len(users_tensor) < 6:
                support_users = torch.from_numpy(np.random.choice(users, self.support_n_samples, replace=True)).long()
                mask = torch.ones(self.support_n_samples)
            else:
                support_users = users_tensor[sampled_indices][self.query_n_samples:]
                n_support = len(support_users)
                support_users = torch.cat((support_users, torch.zeros(self.support_n_samples - n_support, dtype=torch.long)))
                mask = torch.cat((torch.ones(n_support), torch.zeros(self.support_n_samples - n_support)))
        else:
            support_users = users_tensor[sampled_indices][self.query_n_samples:self.query_n_samples + self.support_n_samples]
            mask = torch.ones(10)
        return support_users, query_users, idx, mask

    # TODO: Need to add a condition to only include items that have at least one entry in train set.
    def _load_train_data(self):
        # Load train UI data as a sparse U-I matrix.
        path_ui = os.path.join(self.data_dir, 'popular_train.csv')
        path_val_tr = os.path.join(self.data_dir, 'val.csv')
        path_test_tr = os.path.join(self.data_dir, 'test.csv')

        df_val_tr = pd.read_csv(path_val_tr, header=0, usecols=['item', 'user_0'])
        df_test_tr = pd.read_csv(path_test_tr, header=0, usecols=['item', 'user_0'])
        df_val_tr.rename({'user_0': 'user'}, axis=1, inplace=True)
        df_test_tr.rename({'user_0': 'user'}, axis=1, inplace=True)
        df_ui = pd.read_csv(path_ui)
        rows_ui_popular, cols_ui_popular = df_ui['item'], df_ui['user']
        self.train_data_ui_popular = sparse.csr_matrix((np.ones_like(rows_ui_popular),
                                     (rows_ui_popular, cols_ui_popular)), dtype='float32',
                                    shape=(df_ui['item'].max() + 1, int(df_ui['user'].max() + 1)))


        df_ui = df_ui.append(df_val_tr.groupby('item').apply(lambda x: x[:int(0.7 * len(x))]))  # Take num_tr users for each item
        df_ui = df_ui.append(df_test_tr.groupby('item').apply(lambda x: x[:int(0.7 * len(x))]))  # Take num_tr users for each item
        self.df = df_ui
        self.n_items = df_ui['item'].max() + 1
        self.n_users = int(df_ui['user'].max() + 1)
        rows_ui, cols_ui = df_ui['item'], df_ui['user']

        data_ui = sparse.csr_matrix((np.ones_like(rows_ui),
                                     (rows_ui, cols_ui)), dtype='float32',
                                    shape=(self.n_items, self.n_users))
        return data_ui


class EvalDatasetPopularProNet(data.Dataset):
    """
    load val/test dataset
    """

    def __init__(self, dataset, train_data_ui,
                 datatype='test'):
        """
        Initialize the EvalDataSet object
        Args:
        - dataset: the filename of the dataset used to train, stored in data folder
        - datatype: type of dataset: val or test
        """
        self.dataset = dataset
        self.n_items = train_data_ui.shape[0]
        self.n_users = train_data_ui.shape[1]
        self.support_n_samples = 40
        self.data_dir = os.path.join('../../data/', dataset + "_processed_item")
        self.support_ui_list, self.query_ui = self._load_eval_data(datatype, train_data_ui)
        self.support_ui_lil = [x.tolil() for x in self.support_ui_list]
        self.query_ui_lil = self.query_ui.tolil()

    def __len__(self):
        return self.n_items

    def __getitem__(self, idx):
        """
        idx: a tensor with size (1, ) contains a user index
        @return: [(S_k)], Q_k. A list of tuple of sampled items
                 for this sampled user in each support/query set.
                 We have a fixed query set to evaluate popular users.
        Note we do not shuffle the item set here (to maintain consistency across results).
        We have already shuffled the data set(multiple item_i columns for each user)
        This is to maintain consistency across different models(NCF/VAE-CF).
        """
        support_users = torch.LongTensor(self.support_ui_lil[0].rows[idx])
        n_support = support_users.shape[0]
        if support_users.shape[0] < self.support_n_samples:
            support_users = torch.LongTensor(torch.cat(
                (support_users, torch.zeros(self.support_n_samples - n_support, dtype=torch.long))))
            mask = torch.cat(
                (torch.ones(n_support), torch.zeros(self.support_n_samples - n_support)))
        else:
            support_users = torch.LongTensor(
                np.random.choice(self.support_ui_lil[0].rows[idx], self.support_n_samples))
            mask = torch.ones(self.support_n_samples)

        query_users = torch.LongTensor(self.query_ui_lil.rows[idx])
        query_user_vectors = indices2torch_sparse(query_users, self.n_users).to_dense().squeeze()
        # Binary vector.
        fold_in = self.train_data_ui[idx].toarray().squeeze()
        return support_users, query_user_vectors, fold_in, idx, mask

    def _load_eval_data(self, datatype, train_data_ui):
        # Load train UI data as a sparse U-I matrix for the "unseen" users.
        # Return multiple support sets and a query set.
        self.item_list = range(0, self.n_items)
        self.train_data_ui = train_data_ui
        eval_path_ui = os.path.join(self.data_dir, 'popular_{}.csv'.format(datatype))
        df_eval_ui = pd.read_csv(eval_path_ui)
        rows_query, cols_query = df_eval_ui['item'], df_eval_ui['user']
        query = sparse.csr_matrix((np.ones_like(rows_query),
                                   (rows_query, cols_query)), dtype='float32',
                                  shape=(self.n_items, self.n_users))

        support = []  # List of sparse matrices
        rows_ui_support, cols_ui_support = [], []
        for it in range(self.n_items):
            lst = self.train_data_ui[it].nonzero()[1]
            rows_ui_support.append([it] * len(lst))
            cols_ui_support.append(lst)
        rows_ui_support, cols_ui_support = np.concatenate(rows_ui_support), np.concatenate(cols_ui_support)
        assert rows_ui_support.shape == cols_ui_support.shape

        # Pick 5 users per item as support set.
        support.append(sparse.csr_matrix((np.ones_like(rows_ui_support),
                                          (rows_ui_support, cols_ui_support)), dtype='float32',
                                         shape=(self.n_items, self.n_users)))
        return support, query



class EvalDatasetFewShotProNet(data.Dataset):
    """
    load val/test dataset
    """

    def __init__(self, dataset, n_users, datatype='test',
                 train_data_ui=None):
        """
        Initialize the EvalDataSet object
        Args:
        - dataset: the filename of the dataset used to train, stored in data folder
        - datatype: type of dataset: val or test
        """
        self.dataset = dataset
        self.n_users = n_users
        self.train_data_ui = train_data_ui
        self.n_train_items = self.train_data_ui.shape[0]
        self.data_dir = os.path.join('../../data/', dataset + "_processed_item")
        self.support_ui, self.query_ui = self._load_eval_data(datatype)
        self.n_items = len(self.item_list)
        self.support_ui_lil = [x.tolil() for x in self.support_ui]
        self.query_ui_lil = [x.tolil() for x in self.query_ui]


    def __len__(self):
        return self.n_items

    def __getitem__(self, idx):
        """
        idx: a tensor with size (1, ) contains a user index
        @return: [(S_k, Q_k)...] A list of tuple of sampled items
                 for this sampled user in each support/query set
        Note we do not shuffle the item set here (to maintain consistency across results).
        We have already shuffled the data set(multiple item_i columns for each user)
        This is to maintain consistency across different models(NCF/VAE-CF)
        """


        temp_support = torch.LongTensor(self.support_ui_lil[0].rows[idx])
        support_users = [torch.cat((temp_support, torch.zeros(self.max_training - len(temp_support), dtype=torch.long)))]

        query_users = [torch.LongTensor(self.query_ui_lil[0].rows[idx])]
        query_user_vectors = [indices2torch_sparse(x, self.n_users).to_dense().squeeze() for x in query_users]  #
        # Binary vector.
        samples = torch.cat((torch.ones(len(temp_support)), torch.zeros(self.max_training - len(temp_support))))

        return torch.stack(support_users), torch.stack(query_user_vectors), samples, idx


    def _load_eval_data(self, datatype):
        # Load train UI data as a sparse U-I matrix for the "unseen" users.
        path_ui = os.path.join(self.data_dir, datatype + '.csv')
        df_ui = pd.read_csv(path_ui)
        start_idx, end_idx = df_ui['item'].min(), df_ui['item'].max()
        self.start_idx = start_idx
        self.item_list = range(0, end_idx - start_idx + 1)
        self.items = range(end_idx, start_idx)

        # TODO: later change to numpy operations later (much faster)
        ui_df_support = df_ui.groupby('item').apply(lambda x: x[:int(0.7 * len(x))])
        self.max_training = int(0.7 * df_ui.groupby('item').size().max())

        ui_df_query = df_ui[~df_ui.index.isin(ui_df_support.index.get_level_values(1))]

        assert len(ui_df_support) + len(ui_df_query) == len(df_ui)
        rows_ui_support = ui_df_support['item'] - start_idx
        rows_ui_query = ui_df_query['item'] - start_idx

        user_list = list(df_ui.columns)
        user_list.remove('item')
        support, query, self.query_users = [], [], []
        for user in user_list:
            # For each permutation, create support and query sparse matrices.
            cols_ui_support = ui_df_support[user]
            cols_ui_query = ui_df_query[user]
            self.query_users.append(np.unique(cols_ui_query.values))
            support.append(sparse.csr_matrix((np.ones_like(rows_ui_support),
                                              (rows_ui_support, cols_ui_support)), dtype='float32',
                                             shape=(end_idx - start_idx + 1, self.n_users)))
            query.append(sparse.csr_matrix((np.ones_like(rows_ui_query),
                                            (rows_ui_query, cols_ui_query)), dtype='float32',
                                           shape=(end_idx - start_idx + 1, self.n_users)))

        return support, query



def get_target(df, index, query_users):
    target = df[index].toarray().T
    target = target[query_users]
    return torch.tensor(target)


def get_pos_weight(targets):
    result = (targets == 0).sum(-1).float() / (targets == 1).sum(-1).float()
    return result


class TrainDataset_popular(data.Dataset):
    '''
    Load dataset
    '''

    def __init__(self, dataset):
        self.dataset = dataset
        self.data_dir = os.path.join('../../data/', dataset + "_processed_item")
        self.train_data_ui = self._load_train_data()
        self.user_list = list(range(self.n_users))

    def __len__(self):
        return len(self.user_list)

    def __getitem__(self, index):
        # user = self.user_list[index]
        data = torch.Tensor(self.train_data_ui[index, :].toarray()).squeeze()
        return user, data

    # TODO: Need to add a condition to only include items that have at least one entry in train set.
    def _load_train_data(self):
        # Load train UI data as a sparse U-I matrix.
        path_ui = os.path.join(self.data_dir, 'popular_train.csv')
        path_val_tr = os.path.join(self.data_dir, 'val.csv')
        path_test_tr = os.path.join(self.data_dir, 'test.csv')

        df_val_tr = pd.read_csv(path_val_tr, header=0, usecols=['item', 'user_0'])
        df_test_tr = pd.read_csv(path_test_tr, header=0, usecols=['item', 'user_0'])
        df_val_tr.rename({'user_0': 'user'}, axis=1, inplace=True)
        df_test_tr.rename({'user_0': 'user'}, axis=1, inplace=True)
        df_ui = pd.read_csv(path_ui)
        df_ui = df_ui.append(
            df_val_tr.groupby('item').apply(lambda x: x[:int(0.7 * len(x))]))  # Take num_tr users for each item
        df_ui = df_ui.append(
            df_test_tr.groupby('item').apply(lambda x: x[:int(0.7 * len(x))]))  # Take num_tr users for each item

        self.n_users = df_ui['user'].max() + 1
        self.n_items = int(df_ui['item'].max() + 1)
        rows_ui, cols_ui = df_ui['user'], df_ui['item']

        data_ui = sparse.csr_matrix((np.ones_like(rows_ui),
                                     (rows_ui, cols_ui)), dtype='float32',
                                    shape=(self.n_users, self.n_items))

        return data_ui
