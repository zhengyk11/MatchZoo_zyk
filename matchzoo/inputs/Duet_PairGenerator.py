import os
import random
import time
import numpy as np
import scipy.sparse as sp

from matchzoo import convert_term2id


class Duet_PairGenerator():
    def __init__(self, config):
        self.__name = 'Duet_PairGenerator'

        self.query_maxlen = config['query_maxlen']
        self.doc_maxlen   = config['doc_maxlen']
        self.batch_size = config['batch_size']
        self.data_path = config['data_path']
        self.word_dict = config['word_dict']
        self.ngraph_size = config['ngraph_size']
        self.ngraph = config['ngraph']

        self.rel_gap = 0.
        if 'rel_gap' in config:
            self.rel_gap = config['rel_gap']

        self.high_label = 0.
        if 'high_label' in config:
            self.high_label = config['high_label']

        self.batch_per_iter = config['batch_per_iter']
        self.query_per_iter = config['query_per_iter']
        self.qfile_list = self.get_qfile_list()

        print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        print '[Duet_PairGenerator] init done'

    def get_qfile_list(self):
        qfile_list = []
        for dirpath, dirnames, filenames in os.walk(self.data_path):
            for fn in filenames:
                if fn.endswith('.txt'):
                    qfile_list.append(os.path.join(dirpath, fn))
        return qfile_list

    def get_data(self):
        # while True:
        uid_doc = {}
        qid_query = {}
        qid_label_uid = {}

        qfiles = random.sample(self.qfile_list, self.query_per_iter)
        for fn in qfiles:
            with open(fn) as file:
                for line in file:
                    qid, query, uid, doc, label = line.split('\t')
                    qid = qid.strip()
                    uid = uid.strip()
                    query = convert_term2id(query.strip().split(), self.word_dict)
                    doc = convert_term2id(doc.strip().split(), self.word_dict)
                    label = float(label)
                    qid_query[qid] = query
                    uid_doc[uid] = doc
                    if qid not in qid_label_uid:
                        qid_label_uid[qid] = {}
                    if label not in qid_label_uid[qid]:
                        qid_label_uid[qid][label] = []
                    qid_label_uid[qid][label].append(uid)

        return qid_query, uid_doc, qid_label_uid

    def make_pair(self):
        qid_query, uid_doc, qid_label_uid = self.get_data()
        pair_list = []
        for qid in qid_label_uid:
            for hl in qid_label_uid[qid]:
                for ll in qid_label_uid[qid]:
                    if hl <= ll:
                        continue
                    if hl < self.high_label:
                        continue
                    if hl - ll <= self.rel_gap:
                        continue
                    for dp in qid_label_uid[qid][hl]:
                        for dn in qid_label_uid[qid][ll]:
                            pair_list.append([qid, dp, dn])

        return qid_query, uid_doc, qid_label_uid, pair_list


    def get_batch(self):
        while True:
            qid_query, uid_doc, qid_label_uid, pair_list = self.make_pair()
            print '[%s]' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            print 'Pair Instance Count:', len(pair_list)
            for _i in range(self.batch_per_iter):
                sample_pair_list = random.sample(pair_list, self.batch_size)
                Local_feat = np.zeros((self.batch_size * 2, self.query_maxlen, self.doc_maxlen), dtype=np.int32)
                Dist_feat_query, Dist_feat_doc = [], []
                Y = np.zeros((self.batch_size * 2,), dtype=np.int32)
                Y[::2] = 1

                for i in range(self.batch_size):
                    qid, dp_id, dn_id = sample_pair_list[i]
                    query = qid_query[qid]
                    dp = uid_doc[dp_id]
                    dn = uid_doc[dn_id]

                    query_len = min(self.query_maxlen, len(query))
                    dp_len = min(self.doc_maxlen, len(dp))
                    dn_len = min(self.doc_maxlen, len(dn))
                    query_feat = [[] for tt in range(self.query_maxlen)]
                    for q_i in range(query_len):
                        query_feat[q_i] = self.ngraph[query[q_i]]
                        for dp_j in range(dp_len):
                            if query[q_i] == dp[dp_j]:
                                Local_feat[i*2, q_i, dp_j] = 1
                        for dn_j in range(dn_len):
                            if query[q_i] == dn[dn_j]:
                                Local_feat[i*2+1, q_i, dn_j] = 1
                    query_feat = self.transfer_feat_dense2sparse(query_feat, self.ngraph_size).toarray()
                    Dist_feat_query.append(query_feat)
                    Dist_feat_query.append(query_feat)
                    dp_feat = [[] for tt in range(self.doc_maxlen)]
                    dn_feat = [[] for tt in range(self.doc_maxlen)]
                    for dp_j in range(dp_len):
                        dp_feat[dp_j] = self.ngraph[dp[dp_j]]
                    for dn_j in range(dn_len):
                        dn_feat[dn_j] = self.ngraph[dn[dn_j]]

                    dp_feat = self.transfer_feat_dense2sparse(dp_feat, self.ngraph_size).toarray()
                    dn_feat = self.transfer_feat_dense2sparse(dn_feat, self.ngraph_size).toarray()
                    Dist_feat_doc.append(dp_feat)
                    Dist_feat_doc.append(dn_feat)

                Dist_feat_query = np.array(Dist_feat_query, dtype=np.float32)
                Dist_feat_doc = np.array(Dist_feat_doc, dtype=np.float32)
                yield Local_feat, Dist_feat_query, Dist_feat_doc, Y

    def get_batch_generator(self):
        for Local_feat, Dist_feat_query, Dist_feat_doc, Y in self.get_batch():
            yield ({'Local_feat': Local_feat, 'Dist_feat_query': Dist_feat_query, 'Dist_feat_doc':Dist_feat_doc}, Y)

    def transfer_feat_dense2sparse(self, dense_feat, ngraph_size):
        data = []
        indices = []
        indptr = [0]
        for feat in dense_feat:
            for val in feat:
                indices.append(val)
                data.append(1)
            indptr.append(indptr[-1] + len(feat))
        return sp.csr_matrix((data, indices, indptr), shape=(len(dense_feat), ngraph_size), dtype="float32")
