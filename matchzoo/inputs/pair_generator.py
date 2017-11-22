# -*- coding: utf-8 -*-

import sys
import random
import numpy as np
import time

from utils.rank_io import *
from layers import DynamicMaxPooling

import scipy.sparse as sp

EPS = 1e-20

class PairBasicGenerator(object):
    def __init__(self, config):
        self.__name = 'PairBasicGenerator'
        self.config = config
        self.data1 = config['data1']
        self.data2 = config['data2']
        if 'rel_gap' in config:
            self.rel_gap = config['rel_gap']
        else:
            self.rel_gap = 0.
        if 'high_label' in config:
            self.high_label = config['high_label']
        else:
            self.high_label = 0.
        self.rel = []
        for key in config:
            if 'relation_file' in key:
                rel_file = config[key]
                self.rel += read_relation(filename=rel_file)
        self.rel = list(set(self.rel))
        # rel_file = config['relation_file']
        # self.rel = read_relation(filename=rel_file)
        self.batch_size = config['batch_size']
        self.check_list = ['relation_file', 'batch_size']
        self.point = 0
        if config['use_iter']:
            self.pair_list_iter = self.make_pair_iter(self.rel)
            self.pair_list = []
        else:
            self.pair_list = self.make_pair_static(self.rel)
            self.pair_list_iter = None

    def check(self):
        for e in self.check_list:
            if e not in self.config:
                print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '[%s] Error %s not in config' % (self.__name, e)
                return False
        return True

    def make_pair_static(self, rel):
        rel_set = {}
        pair_list = []
        for label, d1, d2 in rel:
            if d1 not in self.data1:
                continue
            if d2 not in self.data2:
                continue
            if d1 not in rel_set:
                rel_set[d1] = {}
            if label not in rel_set[d1]:
                rel_set[d1][label] = []
            rel_set[d1][label].append(d2)

        for d1 in rel_set:
            label_list = sorted(rel_set[d1].keys(), reverse = True)
            for hidx, high_label in enumerate(label_list[:-1]):
                for low_label in label_list[hidx+1:]:
                    if high_label < self.high_label:
                        continue
                    if high_label - low_label <= self.rel_gap:
                        continue
                    for high_d2 in rel_set[d1][high_label]:
                        for low_d2 in rel_set[d1][low_label]:
                            pair_list.append( (d1, high_d2, low_d2) )
        print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'Pair Instance Count:', len(pair_list)
        return pair_list

    def make_pair_iter(self, rel):
        rel_set = {}
        pair_list = []
        for label, d1, d2 in rel:
            if d1 not in self.data1:
                continue
            if d2 not in self.data2:
                continue
            if d1 not in rel_set:
                rel_set[d1] = {}
            if label not in rel_set[d1]:
                rel_set[d1][label] = []
            rel_set[d1][label].append(d2)
        
        while True:
            pair_list = []
            rel_set_sample = random.sample(rel_set.keys(), self.config['query_per_iter'])

            for d1 in rel_set_sample:
                label_list = sorted(rel_set[d1].keys(), reverse = True)
                for hidx, high_label in enumerate(label_list[:-1]):
                    for low_label in label_list[hidx+1:]:
                        if high_label < self.high_label:
                            continue
                        if high_label - low_label <= self.rel_gap:
                            continue
                        for high_d2 in rel_set[d1][high_label]:
                            for low_d2 in rel_set[d1][low_label]:
                                pair_list.append( (d1, high_d2, low_d2) )
            if len(pair_list) == 0:
                continue
            print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'Pair Instance Count:', len(pair_list)
            yield pair_list
        
    def get_batch_static(self):
        pass

    def get_batch_iter(self):
        pass

    def get_batch(self):
        if self.config['use_iter']:
            return self.batch_iter.next()
        else:
            return self.get_batch_static()

    def get_batch_generator(self):
        pass

    @property
    def num_pairs(self):
        return len(self.pair_list)

    def reset(self):
        self.point = 0

class PairGenerator(PairBasicGenerator):
    def __init__(self, config):
        super(PairGenerator, self).__init__(config=config)
        self.__name = 'PairGenerator'
        self.config = config
        # self.data1 = config['data1']
        # self.data2 = config['data2']
        self.data1_maxlen = config['text1_maxlen']
        self.data2_maxlen = config['text2_maxlen']
        self.fill_word = config['fill_word']
        self.check_list.extend(['data1', 'data2', 'text1_maxlen', 'text2_maxlen', 'fill_word'])
        if config['use_iter']:
            self.batch_iter = self.get_batch_iter()
        if not self.check():
            raise TypeError('[PairGenerator] parameter check wrong.')
        print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '[PairGenerator] init done'

    def get_batch_static(self):
        X1 = np.zeros((self.batch_size*2, self.data1_maxlen), dtype=np.int32)
        X1_len = np.zeros((self.batch_size*2,), dtype=np.int32)
        X2 = np.zeros((self.batch_size*2, self.data2_maxlen), dtype=np.int32)
        X2_len = np.zeros((self.batch_size*2,), dtype=np.int32)
        Y = np.zeros((self.batch_size*2,), dtype=np.int32)

        Y[::2] = 1
        X1[:] = self.fill_word
        X2[:] = self.fill_word
        for i in range(self.batch_size):
            d1, d2p, d2n = random.choice(self.pair_list)
            d1_len = min(self.data1_maxlen, len(self.data1[d1]))
            d2p_len = min(self.data2_maxlen, len(self.data2[d2p]))
            d2n_len = min(self.data2_maxlen, len(self.data2[d2n]))
            X1[i*2,   :d1_len],  X1_len[i*2]   = self.data1[d1][:d1_len],   d1_len
            X2[i*2,   :d2p_len], X2_len[i*2]   = self.data2[d2p][:d2p_len], d2p_len
            X1[i*2+1, :d1_len],  X1_len[i*2+1] = self.data1[d1][:d1_len],   d1_len
            X2[i*2+1, :d2n_len], X2_len[i*2+1] = self.data2[d2n][:d2n_len], d2n_len
            
        return X1, X1_len, X2, X2_len, Y    

    def get_batch_iter(self):
        while True:
            self.pair_list = self.pair_list_iter.next()
            for _ in range(self.config['batch_per_iter']):
                X1 = np.zeros((self.batch_size*2, self.data1_maxlen), dtype=np.int32)
                X1_len = np.zeros((self.batch_size*2,), dtype=np.int32)
                X2 = np.zeros((self.batch_size*2, self.data2_maxlen), dtype=np.int32)
                X2_len = np.zeros((self.batch_size*2,), dtype=np.int32)
                Y = np.zeros((self.batch_size*2,), dtype=np.int32)

                Y[::2] = 1
                X1[:] = self.fill_word
                X2[:] = self.fill_word
                for i in range(self.batch_size):
                    d1, d2p, d2n = random.choice(self.pair_list)
                    d1_len = min(self.data1_maxlen, len(self.data1[d1]))
                    d2p_len = min(self.data2_maxlen, len(self.data2[d2p]))
                    d2n_len = min(self.data2_maxlen, len(self.data2[d2n]))
                    X1[i*2,   :d1_len],  X1_len[i*2]   = self.data1[d1][:d1_len],   d1_len
                    X2[i*2,   :d2p_len], X2_len[i*2]   = self.data2[d2p][:d2p_len], d2p_len
                    X1[i*2+1, :d1_len],  X1_len[i*2+1] = self.data1[d1][:d1_len],   d1_len
                    X2[i*2+1, :d2n_len], X2_len[i*2+1] = self.data2[d2n][:d2n_len], d2n_len
                    
                yield X1, X1_len, X2, X2_len, Y

    def get_batch_generator(self):
        while True:
            X1, X1_len, X2, X2_len, Y = self.get_batch()
            if self.config['use_dpool']:
                yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len, 'dpool_index': DynamicMaxPooling.dynamic_pooling_index(X1_len, X2_len, self.config['text1_maxlen'], self.config['text2_maxlen'])}, Y)
            else:
                yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len}, Y)

class DSSM_PairGenerator(PairBasicGenerator):
    def __init__(self, config):
        super(DSSM_PairGenerator, self).__init__(config=config)
        self.__name = 'DSSM_PairGenerator'
        # self.data1 = config['data1']
        # self.data2 = config['data2']
        self.feat_size = config['feat_size']
        self.check_list.extend(['data1', 'data2', 'feat_size'])
        if config['use_iter']:
            self.batch_iter = self.get_batch_iter()
        if not self.check():
            raise TypeError('[DSSM_PairGenerator] parameter check wrong.')
        print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '[DSSM_PairGenerator] init done'

    def transfer_feat_dense2sparse(self, dense_feat):
        data = []
        indices = []
        indptr = [0]
        for feat in dense_feat:
            for val in feat:
                indices.append(val)
                data.append(1)
            indptr.append(indptr[-1] + len(feat))
        return sp.csr_matrix((data, indices, indptr), shape=(len(dense_feat), self.feat_size), dtype="float32")
    def get_batch_static(self):
        #X1 = np.zeros((self.batch_size*2, self.feat_size), dtype=np.float32)
        X1_len = np.zeros((self.batch_size*2,), dtype=np.int32)
        #X2 = np.zeros((self.batch_size*2, self.feat_size), dtype=np.float32)
        X2_len = np.zeros((self.batch_size*2,), dtype=np.int32)
        Y = np.zeros((self.batch_size*2,), dtype=np.int32)

        Y[::2] = 1
        X1, X2 = [], []
        for i in range(self.batch_size):
            d1, d2p, d2n = random.choice(self.pair_list)
            d1_len = len(self.data1[d1])
            d2p_len = len(self.data2[d2p])
            d2n_len = len(self.data2[d2n])
            X1_len[i*2], X1_len[i*2+1]  = d1_len,  d1_len
            X2_len[i*2], X2_len[i*2+1]  = d2p_len, d2n_len
            X1.append(self.data1[d1])
            X1.append(self.data1[d1])
            X2.append(self.data2[d2p])
            X2.append(self.data2[d2n])
        X1 = self.transfer_feat_dense2sparse(X1).toarray()
        X2 = self.transfer_feat_dense2sparse(X2).toarray()
        return X1, X1_len, X2, X2_len, Y

    def get_batch_iter(self):
        while True:
            self.pair_list = self.pair_list_iter.next()
            for _ in range(self.config['batch_per_iter']):
                #X1 = np.zeros((self.batch_size*2, self.feat_num), dtype=np.float32)
                X1_len = np.zeros((self.batch_size*2,), dtype=np.int32)
                #X2 = np.zeros((self.batch_size*2, self.feat_size), dtype=np.float32)
                X2_len = np.zeros((self.batch_size*2,), dtype=np.int32)
                Y = np.zeros((self.batch_size*2,), dtype=np.int32)

                Y[::2] = 1
                X1, X2 = [], []
                for i in range(self.batch_size):
                    d1, d2p, d2n = random.choice(self.pair_list)
                    d1_len = len(self.data1[d1])
                    d2p_len = len(self.data2[d2p])
                    d2n_len = len(self.data2[d2n])
                    X1_len[i*2],  X1_len[i*2+1]   = d1_len, d1_len
                    X2_len[i*2],  X2_len[i*2+1]   = d2p_len, d2n_len
                    X1.append(self.data1[d1])
                    X1.append(self.data1[d1])
                    X2.append(self.data2[d2p])
                    X2.append(self.data2[d2n])
                    
                yield self.transfer_feat_dense2sparse(X1).toarray(), X1_len, self.transfer_feat_dense2sparse(X2).toarray(), X2_len, Y

    def get_batch_generator(self):
        while True:
            X1, X1_len, X2, X2_len, Y = self.get_batch()
            yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len}, Y)

class DRMM_PairGenerator(PairBasicGenerator):
    def __init__(self, config):
        super(DRMM_PairGenerator, self).__init__(config=config)
        self.__name = 'DRMM_PairGenerator'
        # self.data1 = config['data1']
        # self.data2 = config['data2']
        self.data1_maxlen = config['text1_maxlen']
        self.data2_maxlen = config['text2_maxlen']
        self.embed = config['embed']
        self.hist_size = config['hist_size']
        self.fill_word = config['fill_word']
        self.check_list.extend(['data1', 'data2', 'text1_maxlen', 'text2_maxlen', 'embed', 'hist_size', 'fill_word'])
        self.use_hist_feats = False
        if 'hist_feats' in config:
            self.use_hist_feats = True
            self.hist_feats = config['hist_feats']
        # if 'hist_feats_file' in config:
        #     hist_feats = read_features(config['hist_feats_file'])
        #     self.hist_feats = {}
        #     for idx, (label, d1, d2) in enumerate(self.rel):
        #         self.hist_feats[(d1, d2)] = hist_feats[idx]
        #     self.use_hist_feats = True
        if config['use_iter']:
            self.batch_iter = self.get_batch_iter()
        if not self.check():
            raise TypeError('[DRMM_PairGenerator] parameter check wrong.')
        print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '[DRMM_PairGenerator] init done'

    def cal_hist(self, t1, t2, data1_maxlen, hist_size):
        mhist = np.zeros((data1_maxlen, hist_size), dtype=np.float32)
        d1len = len(self.data1[t1]) 
        if self.use_hist_feats:
            assert (t1, t2) in self.hist_feats
            mhist = np.reshape(self.hist_feats[(t1, t2)], (data1_maxlen, hist_size))
            # if d1len < data1_maxlen:
            #     mhist[:d1len, :] = caled_hist[:, :]
            # else:
            #     mhist[:, :] = caled_hist[:data1_maxlen, :]
        else:
            t1_rep = self.embed[self.data1[t1]]
            t2_rep = self.embed[self.data2[t2]]
            for c in range(t1_rep.shape[0]):
                if np.linalg.norm(t1_rep[c]) > EPS:
                    t1_rep[c] = t1_rep[c] / np.linalg.norm(t1_rep[c])
            for c in range(t2_rep.shape[0]):
                if np.linalg.norm(t2_rep[c]) > EPS:
                    t2_rep[c] = t2_rep[c] / np.linalg.norm(t2_rep[c])
            mm = t1_rep.dot(np.transpose(t2_rep))
            for (i,j), v in np.ndenumerate(mm):
                if i >= data1_maxlen:
                    break
                vid = int((v + 1.) / 2. * ( hist_size - 1.))
                mhist[i][vid] += 1.
            mhist += 1.
            mhist = np.log10(mhist)
        return mhist

    def get_batch_static(self):
        X1 = np.zeros((self.batch_size*2, self.data1_maxlen), dtype=np.int32)
        X1_len = np.zeros((self.batch_size*2,), dtype=np.int32)
        X2 = np.zeros((self.batch_size*2, self.data1_maxlen, self.hist_size), dtype=np.float32)
        X2_len = np.zeros((self.batch_size*2,), dtype=np.int32)
        Y = np.zeros((self.batch_size*2,), dtype=np.int32)

        Y[::2] = 1
        X1[:] = self.fill_word
        #X2[:] = self.fill_word
        for i in range(self.batch_size):
            d1, d2p, d2n = random.choice(self.pair_list)
            d1_len = min(self.data1_maxlen, len(self.data1[d1]))
            d2p_len = len(self.data2[d2p])
            d2n_len = len(self.data2[d2n])
            X1[i*2,   :d1_len],  X1_len[i*2]   = self.data1[d1][:d1_len],   d1_len
            X1[i*2+1, :d1_len],  X1_len[i*2+1] = self.data1[d1][:d1_len],   d1_len
            X2[i*2], X2_len[i*2]   = self.cal_hist(d1, d2p, self.data1_maxlen, self.hist_size), d2p_len
            X2[i*2+1], X2_len[i*2+1] = self.cal_hist(d1, d2n, self.data1_maxlen, self.hist_size), d2n_len
            
        return X1, X1_len, X2, X2_len, Y    

    def get_batch_iter(self):
        while True:
            self.pair_list = self.pair_list_iter.next()
            for _ in range(self.config['batch_per_iter']):
                X1 = np.zeros((self.batch_size*2, self.data1_maxlen), dtype=np.int32)
                X1_len = np.zeros((self.batch_size*2,), dtype=np.int32)
                X2 = np.zeros((self.batch_size*2, self.data1_maxlen, self.hist_size), dtype=np.float32)
                X2_len = np.zeros((self.batch_size*2,), dtype=np.int32)
                Y = np.zeros((self.batch_size*2,), dtype=np.int32)

                Y[::2] = 1
                X1[:] = self.fill_word
                #X2[:] = 0.
                for i in range(self.batch_size):
                    d1, d2p, d2n = random.choice(self.pair_list)
                    d1_len = min(self.data1_maxlen, len(self.data1[d1]))
                    d2p_len = len(self.data2[d2p])
                    d2n_len = len(self.data2[d2n])
                    X1[i*2,   :d1_len],  X1_len[i*2]   = self.data1[d1][:d1_len],   d1_len
                    X1[i*2+1, :d1_len],  X1_len[i*2+1] = self.data1[d1][:d1_len],   d1_len
                    X2[i*2], X2_len[i*2]   = self.cal_hist(d1, d2p, self.data1_maxlen, self.hist_size), d2p_len
                    X2[i*2+1], X2_len[i*2+1] = self.cal_hist(d1, d2n, self.data1_maxlen, self.hist_size), d2n_len
                    
                yield X1, X1_len, X2, X2_len, Y

    def get_batch_generator(self):
        while True:
            X1, X1_len, X2, X2_len, Y = self.get_batch()
            yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len}, Y)

class PairGenerator_Feats(PairBasicGenerator):
    def __init__(self, config):
        super(PairGenerator_Feats, self).__init__(config=config)
        self.__name = 'PairGenerator'
        self.config = config
        self.check_list.extend(['data1', 'data2', 'text1_maxlen', 'text2_maxlen', 'fill_word', 'pair_feat_size', 'pair_feat_file', 'idf_file'])
        if not self.check():
            raise TypeError('[PairGenerator] parameter check wrong.')

        # self.data1 = config['data1']
        # self.data2 = config['data2']
        self.data1_maxlen = config['text1_maxlen']
        self.data2_maxlen = config['text2_maxlen']
        self.fill_word = config['fill_word']
        self.pair_feat_size = config['pair_feat_size']
        pair_feats = read_features(config['pair_feat_file'])
        idf_feats = read_embedding(config['idf_file'])
        self.idf_feats = convert_embed_2_numpy(idf_feats, len(idf_feats)+1)
        self.pair_feats = {}
        for idx, (label, d1, d2) in enumerate(self.rel):
            self.pair_feats[(d1, d2)] = pair_feats[idx]
        if config['use_iter']:
            self.batch_iter = self.get_batch_iter()
        print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '[PairGenerator] init done'

    def get_batch_static(self):
        X1 = np.zeros((self.batch_size*2, self.data1_maxlen), dtype=np.int32)
        X1_len = np.zeros((self.batch_size*2,), dtype=np.int32)
        X2 = np.zeros((self.batch_size*2, self.data2_maxlen), dtype=np.int32)
        X2_len = np.zeros((self.batch_size*2,), dtype=np.int32)
        X3 = np.zeros((self.batch_size * 2, self.pair_feat_size), dtype=np.float32)
        X4 = np.zeros((self.batch_size * 2, self.data1_maxlen), dtype=np.float32)
        Y = np.zeros((self.batch_size*2,), dtype=np.int32)

        Y[::2] = 1
        X1[:] = self.fill_word
        X2[:] = self.fill_word
        for i in range(self.batch_size):
            d1, d2p, d2n = random.choice(self.pair_list)
            d1_len = min(self.data1_maxlen, len(self.data1[d1]))
            d2p_len = min(self.data2_maxlen, len(self.data2[d2p]))
            d2n_len = min(self.data2_maxlen, len(self.data2[d2n]))
            X1[i*2,   :d1_len],  X1_len[i*2]   = self.data1[d1][:d1_len],   d1_len
            X2[i*2,   :d2p_len], X2_len[i*2]   = self.data2[d2p][:d2p_len], d2p_len
            X3[i*2,   :self.pair_feat_size]    = self.pair_feats[(d1, d2p)][:self.pair_feat_size]
            X4[i*2,   :d1_len] = self.idf_feats[self.data1[d1][:d1_len]].reshape((-1,))
            X1[i*2+1, :d1_len],  X1_len[i*2+1] = self.data1[d1][:d1_len],   d1_len
            X2[i*2+1, :d2n_len], X2_len[i*2+1] = self.data2[d2n][:d2n_len], d2n_len
            X3[i*2+1, :self.pair_feat_size]    = self.pair_feats[(d1, d2n)][:self.pair_feat_size]
            X4[i*2+1, :d1_len] = self.idf_feats[self.data1[d1][:d1_len]].reshape((-1,))
            
        return X1, X1_len, X2, X2_len, X3, X4, Y    

    def get_batch_iter(self):
        while True:
            self.pair_list = self.pair_list_iter.next()
            for _ in range(self.config['batch_per_iter']):
                X1 = np.zeros((self.batch_size*2, self.data1_maxlen), dtype=np.int32)
                X1_len = np.zeros((self.batch_size*2,), dtype=np.int32)
                X2 = np.zeros((self.batch_size*2, self.data2_maxlen), dtype=np.int32)
                X2_len = np.zeros((self.batch_size*2,), dtype=np.int32)
                X3 = np.zeros((self.batch_size*2, self.pair_feat_size), dtype=np.float32)
                X4 = np.zeros((self.batch_size*2, self.data1_maxlen), dtype=np.int32)
                Y = np.zeros((self.batch_size*2,), dtype=np.int32)

                Y[::2] = 1
                X1[:] = self.fill_word
                X2[:] = self.fill_word
                for i in range(self.batch_size):
                    d1, d2p, d2n = random.choice(self.pair_list)
                    d1_len = min(self.data1_maxlen, len(self.data1[d1]))
                    d2p_len = min(self.data2_maxlen, len(self.data2[d2p]))
                    d2n_len = min(self.data2_maxlen, len(self.data2[d2n]))
                    X1[i*2,   :d1_len],  X1_len[i*2]   = self.data1[d1][:d1_len],   d1_len
                    X2[i*2,   :d2p_len], X2_len[i*2]   = self.data2[d2p][:d2p_len], d2p_len
                    X3[i*2,   :self.pair_feat_size]    = self.pair_feats[(d1, d2p)][:self.pair_feat_size]
                    X4[i*2,   :d1_len] = self.idf_feats[self.data1[d1][:d1_len]].reshape((-1,))
                    X1[i*2+1, :d1_len],  X1_len[i*2+1] = self.data1[d1][:d1_len],   d1_len
                    X2[i*2+1, :d2n_len], X2_len[i*2+1] = self.data2[d2n][:d2n_len], d2n_len
                    X3[i*2+1, :self.pair_feat_size]    = self.pair_feats[(d1, d2n)][:self.pair_feat_size]
                    X4[i*2+1, :d1_len] = self.idf_feats[self.data1[d1][:d1_len]].reshape((-1,))
                    
                yield X1, X1_len, X2, X2_len, X3, X4, Y

    def get_batch_generator(self):
        while True:
            X1, X1_len, X2, X2_len, X3, X4, Y = self.get_batch()
            yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len, 'query_feats': X4, 'pair_feats': X3}, Y)


class Duet_PairGenerator(PairBasicGenerator):
    def __init__(self, config):
        super(Duet_PairGenerator, self).__init__(config=config)
        self.__name = 'Duet_PairGenerator'
        self.config = config
        # self.data1 = config['data1']
        # self.data2 = config['data2']
        self.data1_maxlen = config['text1_maxlen']
        self.data2_maxlen = config['text2_maxlen']
        self.fill_word = config['fill_word']
        # add for duet
        self.ngraphs, self.num_ngraphs = self.load_ngraphs(config['ngraphs_path'])

        self.check_list.extend(['data1', 'data2', 'text1_maxlen', 'text2_maxlen', 'fill_word'])
        if config['use_iter']:
            self.batch_iter = self.get_batch_iter()
        if not self.check():
            raise TypeError('[Duet_PairGenerator] parameter check wrong.')
        print '[%s]' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '[Duet_PairGenerator] init done'

    def load_ngraphs(self, filename):
        ngraphs = {}
        # max_ngraph_len = 0
        with open(filename) as f:
            for line in f:
                line = line.decode('utf-8', 'ignore')
                w, id = line.strip().split('\t')
                ngraphs[w] = int(id) - 1
                # max_ngraph_len = max(max_ngraph_len, len(w))
        return ngraphs, len(ngraphs)

    def get_batch_static(self):
        local_features = np.zeros((self.batch_size * 2, self.data1_maxlen, self.data2_maxlen), dtype=np.int32)
        ngraph_X1 = np.zeros((self.batch_size * 2, self.num_ngraphs, self.data1_maxlen), dtype=np.int32)
        ngraph_X2 = np.zeros((self.batch_size * 2, self.num_ngraphs, self.data2_maxlen), dtype=np.int32)
        ngraph_X1_len = np.zeros((self.batch_size * 2,), dtype=np.int32)
        ngraph_X2_len = np.zeros((self.batch_size * 2,), dtype=np.int32)
        Y = np.zeros((self.batch_size * 2,), dtype=np.int32)

        Y[::2] = 1
        for i in range(self.batch_size):
            d1, d2p, d2n = random.choice(self.pair_list)
            d1_len = min(self.data1_maxlen, len(self.data1[d1]))
            d2p_len = min(self.data2_maxlen, len(self.data2[d2p]))
            d2n_len = min(self.data2_maxlen, len(self.data2[d2n]))

            ngraph_X1_len[i * 2] = d1_len
            ngraph_X2_len[i * 2] = d2p_len
            ngraph_X1_len[i * 2 + 1] = d1_len
            ngraph_X2_len[i * 2 + 1] = d2n_len

            for d1_idx, d1_w in enumerate(self.data1[d1]):
                for d2p_idx, d2p_w in enumerate(self.data2[d2p]):
                    if d1_w == d2p_w:
                        local_features[i * 2, d1_idx, d2p_idx] = 1
                for d2n_idx, d2n_w in enumerate(self.data2[d2n]):
                    if d1_w == d2n_w:
                        local_features[i * 2 + 1, d1_idx, d2n_idx] = 1
            samples = [self.data1[d1], self.data2[d2p], self.data2[d2n]]

            for idx, data in enumerate(samples):
                ngraph_X = ngraph_X1 if idx == 0 else ngraph_X2
                max_words = self.data1_maxlen if idx == 0 else self.data2_maxlen
                for w_idx, word in enumerate(data[:min(len(data), max_words)]):
                    token = '#' + word + '#'
                    token_len = len(token)
                    for i in range(token_len):
                        for j in range(i + 1, token_len + 1):
                            # if i + j > token_len:
                            #     continue
                            token_tmp = token[i:j]
                            # print('toke:\t', token_tmp)
                            ngraph_idx = self.ngraphs.get(token_tmp)
                            if ngraph_idx == None:
                                continue
                            if idx == 0:
                                ngraph_X[i * 2, ngraph_idx, w_idx] += 1
                                ngraph_X[i * 2 + 1, ngraph_idx, w_idx] += 1
                            else:
                                ngraph_X[i * 2 + idx - 1, ngraph_idx, w_idx] += 1

        return local_features, ngraph_X1, ngraph_X1_len, ngraph_X2, ngraph_X2_len, Y

    def get_batch_iter(self):
        while True:
            self.pair_list = self.pair_list_iter.next()
            for _ in range(self.config['batch_per_iter']):
                local_features = np.zeros((self.batch_size * 2, self.data1_maxlen, self.data2_maxlen), dtype=np.int32)
                ngraph_X1 = np.zeros((self.batch_size * 2, self.num_ngraphs, self.data1_maxlen), dtype=np.int32)
                ngraph_X2 = np.zeros((self.batch_size * 2, self.num_ngraphs, self.data2_maxlen), dtype=np.int32)
                ngraph_X1_len = np.zeros((self.batch_size * 2,), dtype=np.int32)
                ngraph_X2_len = np.zeros((self.batch_size * 2,), dtype=np.int32)
                Y = np.zeros((self.batch_size * 2,), dtype=np.int32)

                Y[::2] = 1
                for i in range(self.batch_size):
                    d1, d2p, d2n = random.choice(self.pair_list)
                    d1_len = min(self.data1_maxlen, len(self.data1[d1]))
                    d2p_len = min(self.data2_maxlen, len(self.data2[d2p]))
                    d2n_len = min(self.data2_maxlen, len(self.data2[d2n]))

                    ngraph_X1_len[i * 2] = d1_len
                    ngraph_X2_len[i * 2] = d2p_len
                    ngraph_X1_len[i * 2 + 1] = d1_len
                    ngraph_X2_len[i * 2 + 1] = d2n_len

                    for d1_idx, d1_w in enumerate(self.data1[d1]):
                        for d2p_idx, d2p_w in enumerate(self.data2[d2p]):
                            if d1_w == d2p_w:
                                local_features[i * 2, d1_idx, d2p_idx] = 1
                        for d2n_idx, d2n_w in enumerate(self.data2[d2n]):
                            if d1_w == d2n_w:
                                local_features[i * 2 + 1, d1_idx, d2n_idx] = 1
                    samples = [self.data1[d1], self.data2[d2p], self.data2[d2n]]

                    for idx, data in enumerate(samples):
                        ngraph_X = ngraph_X1 if idx == 0 else ngraph_X2
                        max_words = self.data1_maxlen if idx == 0 else self.data2_maxlen
                        for w_idx, word in enumerate(data[:min(len(data), max_words)]):
                            token = '#' + word + '#'
                            token_len = len(token)
                            for i in range(token_len):
                                for j in range(i+1, token_len + 1):
                                    # if i + j > token_len:
                                    #     continue
                                    token_tmp = token[i:j]
                                    # print('toke:\t', token_tmp)
                                    ngraph_idx = self.ngraphs.get(token_tmp)
                                    if ngraph_idx == None:
                                        continue
                                    if idx == 0:
                                        ngraph_X[i * 2, ngraph_idx, w_idx] += 1
                                        ngraph_X[i * 2 + 1, ngraph_idx, w_idx] += 1
                                    else:
                                        ngraph_X[i * 2 + idx - 1, ngraph_idx, w_idx] += 1

                yield local_features, ngraph_X1, ngraph_X1_len, ngraph_X2, ngraph_X2_len, Y

    def get_batch_generator(self):
        while True:
            local_features, X1, X1_len, X2, X2_len, Y = self.get_batch()
            # if self.config['use_dpool']:
            #     yield ({'local_features': local_features, 'query': X1, 'query_len': X1_len, 'doc': X2,'doc_len': X2_len,
            #             'dpool_index': DynamicMaxPooling.dynamic_pooling_index(X1_len, X2_len,
            #                                                                    self.config['text1_maxlen'],
            #                                                                    self.config['text2_maxlen'])}, Y)
            # else:
            yield ({'local_features': local_features, 'query': X1, 'query_len': X1_len, 'doc': X2,'doc_len': X2_len}, Y)

