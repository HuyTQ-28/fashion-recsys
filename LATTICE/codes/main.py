import datetime
import math
import os
import random
import sys
from time import time
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.sparse as sparse

from utility.parser import parse_args
from Models import LATTICE
from utility.batch_test import *

args = parse_args()


class Trainer(object):
    def __init__(self, data_config):
        # argument settings
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.model_name = args.model_name
        self.mess_dropout = eval(args.mess_dropout)
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.weight_size = eval(args.weight_size)
        self.n_layers = len(self.weight_size)
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.Ks = eval(args.Ks)
        if args.valid_k in self.Ks:
            self.valid_k_idx = self.Ks.index(args.valid_k)
        else:
            self.valid_k_idx = len(self.Ks) - 1

        self.norm_adj = data_config['norm_adj']
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().cuda()
        
        image_feats, text_feats = self.load_features(data_generator.path)

        self.model = LATTICE(self.n_users, self.n_items, self.emb_dim, self.weight_size, self.mess_dropout,
                             image_feats, text_feats, cache_dir=data_generator.path)
        self.model = self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = self.set_lr_scheduler()

    def load_features(self, dataset_path):
        image_file = os.path.join(dataset_path, 'image_feat.npy')
        text_file = os.path.join(dataset_path, 'text_feat.npy')
        if not os.path.exists(image_file):
            raise FileNotFoundError('image_feat.npy not found in %s' % dataset_path)

        image_feats = np.load(image_file)
        if os.path.exists(text_file):
            text_feats = np.load(text_file)
        else:
            print('text_feat.npy not found in %s; reusing image features for the text branch.' % dataset_path)
            text_feats = image_feats.copy()

        image_feats = self.align_feature_rows(image_feats, 'image_feat.npy')
        text_feats = self.align_feature_rows(text_feats, 'text_feat.npy')
        return image_feats.astype(np.float32), text_feats.astype(np.float32)

    def align_feature_rows(self, feats, name):
        if feats.shape[0] == self.n_items:
            return feats
        if feats.shape[0] > self.n_items:
            return feats[:self.n_items]

        pad_shape = (self.n_items - feats.shape[0], feats.shape[1])
        print('%s has %d rows but data has %d items; padding missing rows with zeros.' %
              (name, feats.shape[0], self.n_items))
        return np.vstack([feats, np.zeros(pad_shape, dtype=feats.dtype)])

    def set_lr_scheduler(self):
        fac = lambda epoch: 0.96 ** (epoch / 50)
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        return scheduler

    def test(self, users_to_test, is_val):
        self.model.eval()
        with torch.no_grad():
            ua_embeddings, ia_embeddings = self.model(self.norm_adj, build_item_graph=True)
        result = test_torch(ua_embeddings, ia_embeddings, users_to_test, is_val)
        return result

    def train(self):
        training_time_list = []
        loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger, map_loger = [], [], [], [], [], []
        stopping_step = 0
        should_stop = False
        cur_best_pre_0 = 0.

        n_batch = data_generator.n_train // args.batch_size + 1
        best_recall = 0
        for epoch in (range(args.epoch)):
            t1 = time()
            loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
            n_batch = data_generator.n_train // args.batch_size + 1
            f_time, b_time, loss_time, opt_time, clip_time, emb_time = 0., 0., 0., 0., 0., 0.
            sample_time = 0.
            build_item_graph = True
            for idx in (range(n_batch)):
                self.model.train()
                self.optimizer.zero_grad()
                sample_t1 = time()
                users, pos_items, neg_items = data_generator.sample()
                sample_time += time() - sample_t1                                                 
                ua_embeddings, ia_embeddings = self.model(self.norm_adj, build_item_graph=build_item_graph)
                build_item_graph = False
                u_g_embeddings = ua_embeddings[users]
                pos_i_g_embeddings = ia_embeddings[pos_items]
                neg_i_g_embeddings = ia_embeddings[neg_items]


                batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                              neg_i_g_embeddings)

                batch_loss = batch_mf_loss + batch_emb_loss + batch_reg_loss

                batch_loss.backward(retain_graph=True)
                self.optimizer.step()

                loss += float(batch_loss)
                mf_loss += float(batch_mf_loss)
                emb_loss += float(batch_emb_loss)
                reg_loss += float(batch_reg_loss)


            self.lr_scheduler.step()

            del ua_embeddings, ia_embeddings, u_g_embeddings, neg_i_g_embeddings, pos_i_g_embeddings

            if math.isnan(loss) == True:
                print('ERROR: loss is nan.')
                sys.exit()

            perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                epoch, time() - t1, loss, mf_loss, emb_loss)
            training_time_list.append(time() - t1)
            print(perf_str)

            if epoch % args.verbose != 0:
                continue


            t2 = time()
            users_to_test = list(data_generator.test_set.keys())
            users_to_val = list(data_generator.val_set.keys())
            ret = self.test(users_to_val, is_val=True)
            training_time_list.append(t2 - t1)

            t3 = time()

            loss_loger.append(loss)
            rec_loger.append(ret['recall'])
            pre_loger.append(ret['precision'])
            ndcg_loger.append(ret['ndcg'])
            hit_loger.append(ret['hit_ratio'])
            map_loger.append(ret['map'])
            if args.verbose > 0:
                perf_str = 'Epoch %d [%.1fs + %.1fs]:  val==[%.5f=%.5f + %.5f + %.5f], ' \
                           'recall=%s, precision=%s, ndcg=%s, map=%s' % \
                           (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss,
                            np.array2string(ret['recall'], precision=5),
                            np.array2string(ret['precision'], precision=5),
                            np.array2string(ret['ndcg'], precision=5),
                            np.array2string(ret['map'], precision=5))
                print(perf_str)

            if ret['recall'][self.valid_k_idx] > best_recall:
                best_recall = ret['recall'][self.valid_k_idx]
                test_ret = self.test(users_to_test, is_val=False)
                perf_str = 'Epoch %d [%.1fs + %.1fs]: test==[%.5f=%.5f + %.5f + %.5f], ' \
                           'recall=%s, precision=%s, ndcg=%s, map=%s' % \
                           (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss,
                            np.array2string(test_ret['recall'], precision=5),
                            np.array2string(test_ret['precision'], precision=5),
                            np.array2string(test_ret['ndcg'], precision=5),
                            np.array2string(test_ret['map'], precision=5))
                print(perf_str)                
                stopping_step = 0
            elif stopping_step < args.early_stopping_patience:
                stopping_step += 1
                print('#####Early stopping steps: %d #####' % stopping_step)
            else:
                print('#####Early stop! #####')
                break

        print(test_ret)

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1./2*(users**2).sum() + 1./2*(pos_items**2).sum() + 1./2*(neg_items**2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.decay * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu

if __name__ == '__main__':
    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()
    config['norm_adj'] = norm_adj

    trainer = Trainer(data_config=config)
    trainer.train()

