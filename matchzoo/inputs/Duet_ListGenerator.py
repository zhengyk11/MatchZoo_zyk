import os
import random
import time
import numpy as np
import scipy.sparse as sp

from utils import convert_term2id


class Duet_ListGenerator():
    def __init__(self, config):
        self.__name = 'Duet_ListGenerator'

        self.query_maxlen = config['query_maxlen']
        self.doc_maxlen   = config['doc_maxlen']
        self.batch_size = config['batch_size']
        self.data_path = config['data_path']
        self.word_dict = config['word_dict']
        self.ngraph_size = config['ngraph_size']
        self.ngraph = config['ngraph']
        self.qfile_list = self.get_qfile_list()
        self.data_handler = open(self.qfile_list[0])# self.get_data_handler()
        self.qfile_idx = 0

        print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        print '[Duet_ListGenerator] init done'

    def reset(self):
        self.data_handler.close()
        # self.data_handler = open(self.qfile_list[0])
        self.qfile_idx = 0

    def get_qfile_list(self):
        qfile_list = []
        for dirpath, dirnames, filenames in os.walk(self.data_path):
            for fn in filenames:
                if fn.endswith('.txt'):
                    qfile_list.append(os.path.join(dirpath, fn))
        return qfile_list

    def get_batch(self):
        while True:
            Local_feat = np.zeros((self.batch_size, self.query_maxlen, self.doc_maxlen), dtype=np.int32)
            Dist_feat_query, Dist_feat_doc = [], []
            Y = np.zeros((self.batch_size,), dtype=np.int32)
            Y[::2] = 1

            curr_batch = []
            for i in range(self.batch_size):
                line = self.data_handler.readline()
                if line == '':
                    if self.qfile_idx == len(self.qfile_list) - 1:
                        break
                    else:
                        self.qfile_idx += 1
                        self.data_handler.close()
                        self.data_handler = open(self.qfile_list[self.qfile_idx])
                        line = self.data_handler.readline()

                qid, query, doc_id, doc, label = line.strip().split('\t')
                qid = qid.strip()
                doc_id = doc_id.strip()
                label = float(label)
                curr_batch.append([qid, doc_id, label])

                query = convert_term2id(query.strip().split(), self.word_dict)
                doc = convert_term2id(doc.strip().split(), self.word_dict)

                query_len = min(self.query_maxlen, len(query))
                doc_len   = min(self.doc_maxlen,   len(doc))
                query_feat = [[] for tt in range(self.query_maxlen)]
                for q_i in range(query_len):
                    query_feat[q_i] = self.ngraph[query[q_i]]
                    for d_j in range(doc_len):
                        if query[q_i] == doc[d_j]:
                            Local_feat[i, q_i, d_j] = 1
                query_feat = self.transfer_feat_dense2sparse(query_feat, self.ngraph_size).toarray()
                Dist_feat_query.append(query_feat)
                doc_feat = [[] for tt in range(self.doc_maxlen)]
                for d_j in range(doc_len):
                    doc_feat[d_j] = self.ngraph[doc[d_j]]
                doc_feat = self.transfer_feat_dense2sparse(doc_feat, self.ngraph_size).toarray()
                Dist_feat_doc.append(doc_feat)
            if len(curr_batch) < 1:
                break
            Dist_feat_query = np.array(Dist_feat_query, dtype=np.float32)
            Dist_feat_doc = np.array(Dist_feat_doc, dtype=np.float32)
            yield Local_feat, Dist_feat_query, Dist_feat_doc, Y, curr_batch

    def get_batch_generator(self):
        for Local_feat, Dist_feat_query, Dist_feat_doc, Y, curr_batch in self.get_batch():
            yield ({'Local_feat': Local_feat, 'Dist_feat_query': Dist_feat_query, 'Dist_feat_doc': Dist_feat_doc}, Y, curr_batch)

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
