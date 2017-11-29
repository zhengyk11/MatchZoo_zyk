import time
import numpy as np
from PairBasicGenerator import PairBasicGenerator


class PairGenerator(PairBasicGenerator):
    def __init__(self, config):
        super(PairGenerator, self).__init__(config=config)
        self.__name = 'PairGenerator'

        self.data1_maxlen = config['text1_maxlen']
        self.data2_maxlen = config['text2_maxlen']


        print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        print '[PairGenerator] init done'

    def get_batch(self):
        while True:
            X1 = np.zeros((self.batch_size * 2, self.data1_maxlen), dtype=np.int32)
            X2 = np.zeros((self.batch_size * 2, self.data2_maxlen), dtype=np.int32)
            Y  = np.zeros((self.batch_size * 2,), dtype=np.int32)
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

                d1_len = min(self.data1_maxlen, len(query))
                dp_len = min(self.data2_maxlen, len(dp))
                dn_len = min(self.data2_maxlen, len(dn))

                X1[i*2,   :d1_len] = query[:d1_len]
                X1[i*2+1, :d1_len] = query[:d1_len]
                X2[i*2,   :dp_len] = dp[:dp_len]
                X2[i*2+1, :dn_len] = dn[:dn_len]

            yield X1, X2, Y

    def get_batch_generator(self):
        while True:
            X1, X2, Y = self.get_batch()
            yield ({'query': X1, 'doc': X2}, Y)