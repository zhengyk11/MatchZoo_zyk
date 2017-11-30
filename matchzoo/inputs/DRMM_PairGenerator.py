import os
import time
import random
import numpy as np

from matchzoo import convert_term2id


class DRMM_PairGenerator():
    def __init__(self, config):
        self.__name = 'DRMM_PairGenerator'

        self.query_maxlen = config['query_maxlen']
        self.hist_size    = config['hist_size']
        self.batch_size = config['batch_size']
        self.data_path = config['data_path']
        self.word_dict = config['word_dict']

        self.rel_gap = 0.
        if 'rel_gap' in config:
            self.rel_gap = config['rel_gap']

        self.high_label = 0.
        if 'high_label' in config:
            self.high_label = config['high_label']

        self.batch_per_iter = config['batch_per_iter']
        self.query_per_iter = config['query_per_iter']
        self.qfile_list = self.get_qfile_list()

        print '[%s]' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        print '[DRMM_PairGenerator] init done'

    def get_qfile_list(self):
        qfile_list = []
        for dirpath, dirnames, filenames in os.walk(self.data_path):
            for fn in filenames:
                if fn.endswith('.txt'):
                    qfile_list.append(os.path.join(dirpath, fn))
        return qfile_list

    def get_data(self):
        # while True:
        uid_doc = {}
        qid_query = {}
        qid_label_uid = {}

        qfiles = random.sample(self.qfile_list, self.query_per_iter)
        for fn in qfiles:
            with open(fn) as file:
                for line in file:
                    qid, query, uid, doc, label = line.split('\t')
                    qid = qid.strip()
                    uid = uid.strip()
                    query = convert_term2id(query.strip().split(), self.word_dict)
                    doc = map(float, doc.strip().split())
                    # doc = np.array(map(float, doc.strip().split()), dtype=np.float32) # convert_term2id(doc.strip().split(), self.word_dict)
                    # doc = np.reshape(doc, [self.query_maxlen, self.hist_size])
                    label = float(label)
                    qid_query[qid] = query
                    uid_doc[uid] = doc
                    if qid not in qid_label_uid:
                        qid_label_uid[qid] = {}
                    if label not in qid_label_uid[qid]:
                        qid_label_uid[qid][label] = []
                    qid_label_uid[qid][label].append(uid)

        return qid_query, uid_doc, qid_label_uid

    def make_pair(self):
        qid_query, uid_doc, qid_label_uid = self.get_data()
        pair_list = []
        for qid in qid_label_uid:
            for hl in qid_label_uid[qid]:
                for ll in qid_label_uid[qid]:
                    if hl <= ll:
                        continue
                    if hl < self.high_label:
                        continue
                    if hl - ll <= self.rel_gap:
                        continue
                    for dp in qid_label_uid[qid][hl]:
                        for dn in qid_label_uid[qid][ll]:
                            pair_list.append([qid, dp, dn])

        return qid_query, uid_doc, qid_label_uid, pair_list

    def get_batch(self):
        while True:
            qid_query, uid_doc, qid_label_uid, pair_list = self.make_pair()
            print '[%s]' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            print 'Pair Instance Count:', len(pair_list)
            for _i in range(self.batch_per_iter):
                sample_pair_list = random.sample(pair_list, self.batch_size)
                X1 = np.zeros((self.batch_size * 2, self.query_maxlen), dtype=np.int32)
                X2 = np.zeros((self.batch_size * 2, self.query_maxlen, self.hist_size), dtype=np.float32)
                Y  = np.zeros((self.batch_size * 2,), dtype=np.int32)
                Y[::2] = 1

                for i in range(self.batch_size):
                    qid, dp_id, dn_id = sample_pair_list[i]
                    query = qid_query[qid]
                    dp = uid_doc[dp_id]
                    dn = uid_doc[dn_id]

                    dp_hist = np.reshape(dp, (self.query_maxlen, self.hist_size))
                    dn_hist = np.reshape(dn, (self.query_maxlen, self.hist_size))

                    query_len = min(self.query_maxlen, len(query))

                    X1[i * 2, :query_len] = query[:query_len]
                    X1[i * 2 + 1, :query_len] = query[:query_len]
                    X2[i * 2] = dp_hist
                    X2[i * 2 + 1] = dn_hist

                yield X1, X2, Y

    def get_batch_generator(self):
        for X1, X2, Y in self.get_batch():
            yield ({'query': X1, 'doc': X2}, Y)
