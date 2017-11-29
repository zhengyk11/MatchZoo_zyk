import time
import numpy as np
from ListBasicGenerator import ListBasicGenerator


class DRMM_ListGenerator(ListBasicGenerator):
    def __init__(self, config):
        super(DRMM_ListGenerator, self).__init__(config=config)
        self.__name = 'DRMM_ListGenerator'

        self.query_maxlen = config['query_maxlen']
        self.hist_size    = config['hist_size']

        print '[%s]' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        print '[DRMM_ListGenerator] init done'


    def get_batch(self):
        while True:
            X1 = np.zeros((self.batch_size, self.query_maxlen), dtype=np.int32)
            X2 = np.zeros((self.batch_size, self.query_maxlen, self.hist_size), dtype=np.float32)
            Y  = np.zeros((self.batch_size,), dtype=np.int32)
            Y[::2] = 1

            curr_batch = []
            for i in range(self.batch_size):
                line = self.data_handler.readline()
                if line == '':
                    break
                qid, query, doc_id, doc, doc_score = line.strip().split('\t')
                curr_batch.append([qid, doc_id, doc_score])

                query = map(int, query.split())
                doc    = map(float, doc.split())

                doc_hist = np.reshape(doc, (self.query_maxlen, self.hist_size))

                query_len = min(self.query_maxlen, len(query))

                X1[i, :query_len] = query[:query_len]
                X2[i] = doc_hist

            if len(curr_batch) < 1:
                break
            yield X1, X2, Y, curr_batch


    def get_batch_generator(self):
        for X1, X2, Y, curr_batch in self.get_batch():
            yield ({'query': X1, 'doc': X2}, Y, curr_batch)
