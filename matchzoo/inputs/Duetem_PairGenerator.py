import os
import random
import time
import numpy as np

from utils import convert_term2id


class Duetem_PairGenerator():
    def __init__(self, config):
        self.__name = 'Duetem_PairGenerator'

        self.query_maxlen = config['query_maxlen']
        self.doc_maxlen   = config['doc_maxlen']
        self.batch_size = config['batch_size']
        self.data_path = config['data_path']
        self.word_dict = config['word_dict']

        self.label_index = 4
        if 'label_index' in config:
            self.label_index += config['label_index']

        self.rel_gap = 0.
        if 'rel_gap' in config:
            self.rel_gap = config['rel_gap']

        self.high_label = 0.
        if 'high_label' in config:
            self.high_label = config['high_label']

        self.batch_per_iter = config['batch_per_iter']
        self.query_per_iter = config['query_per_iter']
        self.qfile_list = self.get_qfile_list()

        print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        print '[Duetem_PairGenerator] init done'

    def get_qfile_list(self):
        qfile_list = []
        for dirpath, dirnames, filenames in os.walk(self.data_path):
            for fn in filenames:
                if fn.endswith('.txt'):
                    qfile_list.append(os.path.join(dirpath, fn))
        return qfile_list

    def get_data(self):
        uid_doc = {}
        qid_query = {}
        qid_label_uid = {}

        qfiles = random.sample(self.qfile_list, self.query_per_iter)
        for fn in qfiles:
            with open(fn) as file:
                for line in file:
                    # qid, query, uid, doc, label = line.split('\t')
                    # qid = qid.strip()
                    # uid = uid.strip()

                    attr = line.strip().split('\t')
                    qid = attr[0].strip()
                    query = attr[1].strip()
                    uid = attr[2].strip()
                    doc = attr[3].strip()
                    label = float(attr[self.label_index])

                    query = convert_term2id(query.strip().split(), self.word_dict)
                    doc = convert_term2id(doc.strip().split(), self.word_dict)

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
                            pair_list.append([qid, dp, hl, dn, ll])

        return qid_query, uid_doc, qid_label_uid, pair_list


    def get_batch(self):
        while True:
            qid_query, uid_doc, qid_label_uid, pair_list = self.make_pair()
            if len(pair_list) < self.batch_size:
                continue
            print '[%s]' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            print 'Pair Instance Count:', len(pair_list)
            for _i in range(self.batch_per_iter):
                sample_pair_list = random.sample(pair_list, self.batch_size)
                Local_feat = np.zeros((self.batch_size * 2, self.query_maxlen, self.doc_maxlen), dtype=np.int32)
                X1 = np.zeros((self.batch_size * 2, self.query_maxlen), dtype=np.int32)
                X2 = np.zeros((self.batch_size * 2, self.doc_maxlen), dtype=np.int32)
                Y = np.zeros((self.batch_size * 2,), dtype=np.float32)
                # Y[::2] = 1

                for i in range(self.batch_size):
                    qid, dp_id, dp_label, dn_id, dn_label = sample_pair_list[i]
                    query = qid_query[qid]
                    dp = uid_doc[dp_id]
                    dn = uid_doc[dn_id]

                    query_len = min(self.query_maxlen, len(query))
                    dp_len = min(self.doc_maxlen, len(dp))
                    dn_len = min(self.doc_maxlen, len(dn))

                    # generate local feature matrix
                    query_index = {}
                    for query_i in range(query_len):
                        wid = query[query_i]
                        if wid in query_index:
                            query_index[wid].append(query_i)
                        else:
                            query_index[wid] = [query_i]
                    for doc, doc_len, bias in zip([dp, dn], [dp_len, dn_len], [0, 1]):
                        for doc_i in range(doc_len):
                            wid = doc[doc_i]
                            if wid not in query_index:
                                continue
                            q_i_list = query_index[wid]
                            for q_i in q_i_list:
                                Local_feat[i * 2 + bias, q_i, doc_i] = 1

                    X1[i*2,   :query_len] = query[:query_len]
                    X1[i*2+1, :query_len] = query[:query_len]
                    X2[i*2,   :dp_len] = dp[:dp_len]
                    X2[i*2+1, :dn_len] = dn[:dn_len]
                    Y[i*2]   = dp_label
                    Y[i*2+1] = dn_label

                Y = np.reshape(np.exp(Y), [-1, 2])
                Y /= np.sum(Y, axis=1)[:,None]
                Y = np.reshape(Y, [-1])
                # Y[:] = 0
                # Y[::2] = 1
                yield Local_feat, X1, X2, Y

    def get_batch_generator(self):
        for Local_feat, X1, X2, Y in self.get_batch():
            yield ({'local_feat': Local_feat, 'query': X1, 'doc': X2}, Y)

