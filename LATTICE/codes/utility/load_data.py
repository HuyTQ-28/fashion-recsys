import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import json
import csv
import os
from utility.parser import parse_args
args = parse_args()

def resolve_dataset_path(data_path, dataset):
    data_path = os.path.abspath(data_path)
    dataset_path = os.path.join(data_path, dataset) if dataset else data_path
    if os.path.isdir(dataset_path):
        return dataset_path
    return data_path

class Data(object):
    def __init__(self, path, batch_size):
        path = os.path.abspath(path)
        self.path = os.path.join(path, '%d-core' % args.core)
        self.batch_size = batch_size

        train_file = os.path.join(self.path, 'train.json')
        val_file = os.path.join(self.path, 'val.json')
        test_file = os.path.join(self.path, 'test.json')

        #get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_val, self.n_test = 0, 0, 0
        self.neg_pools = {}

        self.exist_users = []

        if os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file):
            train = json.load(open(train_file))
            test = json.load(open(test_file))
            val = json.load(open(val_file))
        else:
            self.path = path
            train, val, test = self._load_freedom_interactions(path)

        for split in (train, val, test):
            for uid, items in split.items():
                if len(items) == 0:
                    continue
                uid = int(uid)
                self.n_users = max(self.n_users, uid)
                self.n_items = max(self.n_items, max(items))

        for uid, items in train.items():
            if len(items) == 0:
                continue
            uid = int(uid)
            self.exist_users.append(uid)
            self.n_train += len(items)

        for uid, items in test.items():
            uid = int(uid)
            try:
                self.n_test += len(items)
            except:
                continue

        for uid, items in val.items():
            uid = int(uid)
            try:
                self.n_val += len(items)
            except:
                continue

        self.n_items += 1
        self.n_users += 1

        self.print_statistics()

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.R_Item_Interacts = sp.dok_matrix((self.n_items, self.n_items), dtype=np.float32)

        self.train_items, self.test_set, self.val_set = {}, {}, {}
        for uid, train_items in train.items():
            if len(train_items) == 0:
                continue
            uid = int(uid)
            for idx, i in enumerate(train_items):
                self.R[uid, i] = 1.

            self.train_items[uid] = train_items

        for uid, test_items in test.items():
            uid = int(uid)
            if len(test_items) == 0:
                continue
            try:
                self.test_set[uid] = test_items
            except:
                continue            

        for uid, val_items in val.items():
            uid = int(uid)
            if len(val_items) == 0:
                continue
            try:
                self.val_set[uid] = val_items
            except:
                continue

    def _load_freedom_interactions(self, path):
        inter_file = self._find_inter_file(path)
        train, val, test = {}, {}, {}
        splits = {0: train, 1: val, 2: test}

        with open(inter_file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            required = ['userID:token', 'itemID:token', 'x_label:float']
            missing = [col for col in required if col not in (reader.fieldnames or [])]
            if missing:
                raise ValueError('%s is missing required columns: %s' % (inter_file, ', '.join(missing)))

            for row in reader:
                uid = int(row['userID:token'])
                iid = int(row['itemID:token'])
                split = int(float(row['x_label:float']))
                if split not in splits:
                    continue
                splits[split].setdefault(uid, []).append(iid)

        if not train:
            raise ValueError('%s does not contain any training interactions with x_label:float == 0' % inter_file)
        return train, val, test

    def _find_inter_file(self, path):
        candidates = []
        if args.dataset:
            candidates.append(os.path.join(path, '%s.inter' % args.dataset))
        candidates.extend(
            os.path.join(path, filename)
            for filename in os.listdir(path)
            if filename.endswith('.inter')
        )
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError('No LATTICE json split or FREEDOM .inter file found in %s' % path)

    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)

        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
        return adj_mat, norm_adj_mat, mean_adj_mat

    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def get_D_inv(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            return d_mat_inv

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()


    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]
        # users = self.exist_users[:]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)
            # neg_items += sample_neg_items_for_u(u, 3)
        return users, pos_items, neg_items



    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_val + self.n_test))
        print('n_train=%d, n_val=%d, n_test=%d, sparsity=%.5f' %
              (self.n_train, self.n_val, self.n_test,
               (self.n_train + self.n_val + self.n_test)/(self.n_users * self.n_items)))

