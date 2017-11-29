import os
import time
import numpy as np
import scipy.sparse as sp
from utils import convert_term2id

class ListGenerator():
    def __init__(self, config):
        self.__name = 'ListGenerator'

        self.query_maxlen = config['query_maxlen']
        self.doc_maxlen   = config['doc_maxlen']
        self.batch_size = config['batch_size']
        self.data_path = config['data_path']
        self.word_dict = config['word_dict']
        self.qfile_list = self.get_qfile_list()
        self.data_handler = open(self.qfile_list[0])# self.get_data_handler()
        self.qfile_idx = 0

        print '[%s]' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        print '[ListGenerator] init done'

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
            X1 = np.zeros((self.batch_size, self.query_maxlen), dtype=np.int32)
            X2 = np.zeros((self.batch_size, self.doc_maxlen),   dtype=np.int32)
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

                query = convert_term2id(query.strip().split(), self.word_dict)
                doc = convert_term2id(doc.strip().split(), self.word_dict)

                # query = map(int, query.split())
                # doc   = map(int, doc.split())

                query_len = min(self.query_maxlen, len(query))
                doc_len   = min(self.doc_maxlen,   len(doc))

                X1[i, :query_len] = query[:query_len]
                X2[i, :doc_len]   = doc[:doc_len]

            if len(curr_batch) < 1:
                break
            yield X1, X2, Y, curr_batch

    def get_batch_generator(self):
        for X1, X2, Y, curr_batch in self.get_batch():
            yield ({'query': X1, 'doc': X2}, Y, curr_batch)