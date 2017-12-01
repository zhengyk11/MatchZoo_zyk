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

                # query = query.strip().decode('utf-8', 'ignore')
                # query = ' '.join([w for w in query]).encode('utf-8', 'ignore')
                # query = convert_term2id(query.strip().split(), self.ngraph)
                # doc = doc.strip().decode('utf-8', 'ignore')
                # doc = ' '.join([w for w in doc]).encode('utf-8', 'ignore')
                # doc = convert_term2id(doc.strip().split(), self.ngraph)
                # query = convert_term2id(query.strip().split(), self.word_dict)
                # doc = convert_term2id(doc.strip().split(), self.word_dict)

                X1.append(query_hash)
                X2.append(doc_hash)

            if len(curr_batch) < 1:
                break
            X1 = np.array(X1, dtype=np.float32)
            X2 = np.array(X2, dtype=np.float32)
            # X1 = self.transfer_feat_dense2sparse(X1).toarray()
            # X2 = self.transfer_feat_dense2sparse(X2).toarray()
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
