import os
import time
import numpy as np
import scipy.sparse as sp
# from ListBasicGenerator import ListBasicGenerator

from utils import convert_term2id


class DSSM_ListGenerator(): #  ListBasicGenerator):
    def __init__(self, config):
        # super(DSSM_ListGenerator, self).__init__(config=config)
        self.__name = 'DSSM_ListGenerator'

        self.feat_size = config['ngraph_size'] # config['feat_size']
        self.ngraph_size = config['ngraph_size']
        self.ngraph = config['ngraph']
        self.query_maxlen = config['query_maxlen']
        self.doc_maxlen   = config['doc_maxlen']
        self.batch_size = config['batch_size']
        self.data_path = config['data_path']
        # self.word_dict = config['word_dict']
        self.qfile_list = self.get_qfile_list()
        self.data_handler = open(self.qfile_list[0])# self.get_data_handler()
        self.qfile_idx = 0

        print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        print '[DSSM_ListGenerator] init done'

    def __del__(self):
        self.data_handler.close()

    def reset(self):
        self.data_handler.close()
        self.data_handler = open(self.qfile_list[0])
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
            X1, X2 = [], []
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
                        self.data_handler = open(self.qfile_list[self.qfile_idx])
                        line = self.data_handler.readline()
                qid, query, doc_id, doc, label = line.strip().split('\t')
                qid = qid.strip()
                doc_id = doc_id.strip()
                label = float(label)
                curr_batch.append([qid, doc_id, label])

                query = query.strip().split()
                query_len = min(len(query), self.query_maxlen)
                query = query[:query_len]
                query = convert_term2id(query, self.ngraph)

                doc = doc.strip().split()
                doc_len = min(len(doc), self.doc_maxlen)
                doc = doc[:doc_len]
                doc = convert_term2id(doc, self.ngraph)

                X1.append(query)
                X2.append(doc)

            if len(curr_batch) < 1:
                break

            X1 = self.transfer_feat_dense2sparse(X1, self.ngraph_size).toarray()
            X2 = self.transfer_feat_dense2sparse(X2, self.ngraph_size).toarray()
            yield X1, X2, Y, curr_batch

    def get_batch_generator(self):
        for X1, X2, Y, curr_batch in self.get_batch():
            yield ({'query': X1, 'doc': X2}, Y, curr_batch)

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
