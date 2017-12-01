import os
import time
import random
import numpy as np
import scipy.sparse as sp

from utils import convert_term2id
# from PairBasicGenerator import PairBasicGenerator


class DSSM_PairGenerator(): # PairBasicGenerator):
    def __init__(self, config):
        # super(DSSM_PairGenerator, self).__init__(config=config)
        self.__name = 'DSSM_PairGenerator'

        self.query_maxlen = config['query_maxlen']
        self.doc_maxlen   = config['doc_maxlen']
        self.batch_size = config['batch_size']
        self.data_path = config['data_path']
        # self.word_dict = config['word_dict']
        self.feat_size = config['ngraph_size'] # config['feat_size']
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
        print '[DSSM_PairGenerator] init done'

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

                    query = query.strip().split()[:min(len(query), self.query_maxlen)]
                    query_hash = [[] for tt in range(self.query_maxlen)]
                    for q_i, query_term in enumerate(query):
                        query_term = query_term.strip().decode('utf-8', 'ignore')
                        query_term = ' '.join([w for w in query_term]).encode('utf-8', 'ignore').strip().split()
                        query_term_id = convert_term2id(query_term, self.ngraph)
                        query_hash[q_i] = query_term_id
                    query_hash = self.transfer_feat_dense2sparse(query_hash, self.ngraph_size)
                    doc = doc.strip().split()[:min(len(doc), self.doc_maxlen)]
                    doc_hash = [[] for tt in range(self.doc_maxlen)]
                    for d_i, doc_term in enumerate(doc):
                        doc_term = doc_term.strip().decode('utf-8', 'ignore')
                        doc_term = ' '.join([w for w in doc_term]).encode('utf-8', 'ignore').strip().split()
                        doc_term_id = convert_term2id(doc_term, self.ngraph)
                        doc_hash[d_i] = doc_term_id
                    doc_hash = self.transfer_feat_dense2sparse(doc_hash, self.ngraph_size)

                    label = float(label)
                    qid_query[qid] = query_hash
                    uid_doc[uid] = doc_hash
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
                X1, X2 = [], []
                Y = np.zeros((self.batch_size * 2,), dtype=np.int32)
                Y[::2] = 1

                for i in range(self.batch_size):
                    qid, dp_id, dn_id = sample_pair_list[i]
                    query = qid_query[qid]
                    dp = uid_doc[dp_id]
                    dn = uid_doc[dn_id]

                    X1.append(query)
                    X1.append(query)
                    X2.append(dp)
                    X2.append(dn)

                X1 = np.array(X1, dtype=np.float32)
                X2 = np.array(X2, dtype=np.float32)
                # X1 = self.transfer_feat_dense2sparse(X1, self.feat_size).toarray()
                # X2 = self.transfer_feat_dense2sparse(X2, self.feat_size).toarray()
                yield X1, X2, Y

    def get_batch_generator(self):
        for X1, X2, Y in self.get_batch():
            yield ({'query': X1, 'doc': X2}, Y)

    def transfer_feat_dense2sparse(self, dense_feat, feat_size):
        data = []
        indices = []
        indptr = [0]
        for feat in dense_feat:
            for val in feat:
                indices.append(val)
                data.append(1)
            indptr.append(indptr[-1] + len(feat))
        return sp.csr_matrix((data, indices, indptr), shape=(len(dense_feat), feat_size), dtype="float32")
