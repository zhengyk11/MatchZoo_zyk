import time
import numpy as np
from PairBasicGenerator import PairBasicGenerator


class DRMM_PairGenerator(PairBasicGenerator):
    def __init__(self, config):
        super(DRMM_PairGenerator, self).__init__(config=config)
        self.__name = 'DRMM_PairGenerator'

        self.data1_maxlen = config['text1_maxlen']
        self.data2_maxlen = config['text2_maxlen']
        self.hist_size = config['hist_size']

        print '[%s]' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        print '[DRMM_PairGenerator] init done'


    def get_batch(self):
        while True:
            X1 = np.zeros((self.batch_size * 2, self.data1_maxlen), dtype=np.int32)
            X2 = np.zeros((self.batch_size * 2, self.data1_maxlen, self.hist_size), dtype=np.float32)
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

                dp_hist = np.reshape(dp, (self.data1_maxlen, self.hist_size))
                dn_hist = np.reshape(dn, (self.data1_maxlen, self.hist_size))

                d1_len = min(self.data1_maxlen, len(query))

                X1[i*2,   :d1_len] = query[:d1_len]
                X1[i*2+1, :d1_len] = query[:d1_len]
                X2[i*2]   = dp_hist
                X2[i*2+1] = dn_hist

            yield X1, X2, Y


    def get_batch_generator(self):
        while True:
            X1, X2, Y = self.get_batch()
            yield ({'query': X1, 'doc': X2}, Y)
