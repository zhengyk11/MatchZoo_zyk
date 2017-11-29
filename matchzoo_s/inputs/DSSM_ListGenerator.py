import time
import numpy as np
import scipy.sparse as sp
from ListBasicGenerator import ListBasicGenerator


class DSSM_ListGenerator(ListBasicGenerator):
    def __init__(self, config):
        super(DSSM_ListGenerator, self).__init__(config=config)
        self.__name = 'DSSM_ListGenerator'

        self.feat_size = config['feat_size']

        print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        print '[DSSM_ListGenerator] init done'

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


    def get_batch(self):
        while True:
            X1, X2 = [], []
            Y = np.zeros((self.batch_size*2,), dtype=np.int32)
            Y[::2] = 1

            curr_batch = []
            for i in range(self.batch_size):
                line = self.data_handler.readline()
                if line == '':
                    break
                qid, query, doc_id, doc, doc_score = line.strip().split('\t')
                curr_batch.append([qid, doc_id, doc_score])

                query = map(int, query.split())
                doc   = map(int, doc.split())

                X1.append(query)
                X2.append(doc)

            if len(curr_batch) < 1:
                break

            X1 = self.transfer_feat_dense2sparse(X1).toarray()
            X2 = self.transfer_feat_dense2sparse(X2).toarray()
            yield X1, X2, Y, curr_batch

    def get_batch_generator(self):
        for X1, X2, Y, curr_batch in self.get_batch():
            yield ({'query': X1, 'doc': X2}, Y, curr_batch)