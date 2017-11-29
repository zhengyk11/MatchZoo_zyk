import time
import numpy as np


class DRMM_PairGenerator():
    def __init__(self, config):
        self.__name = 'DRMM_PairGenerator'

        self.query_maxlen = config['query_maxlen']
        self.hist_size    = config['hist_size']
        self.batch_size = config['batch_size']

        print '[%s]' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        print '[DRMM_PairGenerator] init done'


    def get_batch(self):
        while True:
            X1 = np.zeros((self.batch_size * 2, self.query_maxlen), dtype=np.int32)
            X2 = np.zeros((self.batch_size * 2, self.query_maxlen, self.hist_size), dtype=np.float32)
            Y  = np.zeros((self.batch_size * 2,), dtype=np.int32)
            Y[::2] = 1

            for i in range(self.batch_size):
                line = self.data_handler.readline()
                if line == '':
                    self.data_handler.seek(0)
                    line = self.data_handler.readline()
                qid, query, dp_id, dp, dp_score, dn_id, dn, dn_score = line.strip().split('\t')

                query = map(int, query.split())
                dp    = map(float, dp.split())
                dn    = map(float, dn.split())

                dp_hist = np.reshape(dp, (self.query_maxlen, self.hist_size))
                dn_hist = np.reshape(dn, (self.query_maxlen, self.hist_size))

                query_len = min(self.query_maxlen, len(query))

                X1[i*2,   :query_len] = query[:query_len]
                X1[i*2+1, :query_len] = query[:query_len]
                X2[i*2]   = dp_hist
                X2[i*2+1] = dn_hist

            yield X1, X2, Y


    def get_batch_generator(self):
        while True:
            X1, X2, Y = self.get_batch()
            yield ({'query': X1, 'doc': X2}, Y)
