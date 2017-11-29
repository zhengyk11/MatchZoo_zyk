import time
import numpy as np
import scipy.sparse as sp
from PairBasicGenerator import PairBasicGenerator


class DSSM_PairGenerator(PairBasicGenerator):
    def __init__(self, config):
        super(DSSM_PairGenerator, self).__init__(config=config)
        self.__name = 'DSSM_PairGenerator'

        self.feat_size = config['feat_size']

        print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        print '[DSSM_PairGenerator] init done'

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

            for i in range(self.batch_size):
                line = self.data_handler.readline()
                if line == '':
                    self.data_handler.seek(0)
                    line = self.data_handler.readline()
                qid, query, dp_id, dp, dp_score, dn_id, dn, dn_score = line.strip().split('\t')

                query = map(int, query.split())
                dp    = map(int, dp.split())
                dn    = map(int, dn.split())

                X1.append(query)
                X1.append(query)
                X2.append(dp)
                X2.append(dn)

            X1 = self.transfer_feat_dense2sparse(X1).toarray()
            X2 = self.transfer_feat_dense2sparse(X2).toarray()
            yield X1, X2, Y

    def get_batch_generator(self):
        while True:
            X1, X2, Y = self.get_batch()
            yield ({'query': X1, 'doc': X2}, Y)