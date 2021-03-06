import os
import time
import numpy as np

from utils import convert_term2id


class Duetem_ListGenerator():
    def __init__(self, config):
        self.__name = 'Duetem_ListGenerator'

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

        self.qfile_list = self.get_qfile_list()
        assert len(self.qfile_list) > 0
        self.data_handler = open(self.qfile_list[0])

        self.data, self.qid_rel_uid = self.get_all_batch()
        self.all_pairs = self.make_pairs()
        self.data_handler.close()

        print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        print '[Duetem_ListGenerator] init done'

    def get_all_pairs(self):
        return self.all_pairs

    def make_pairs(self):
        all_pairs = {}
        for qid in self.qid_rel_uid:
            for hr in self.qid_rel_uid[qid]:
                for lr in self.qid_rel_uid[qid]:
                    if hr <= lr:
                        continue
                    if hr - lr <= self.rel_gap:
                        continue
                    if hr < self.high_label:
                        continue
                    for huid in self.qid_rel_uid[qid][hr]:
                        for luid in self.qid_rel_uid[qid][lr]:
                            all_pairs[(qid, huid, luid)] = [hr, lr]
        print '[%s]' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        print '[ListGenerator] Pair Instance Count:', len(all_pairs)
        return all_pairs

    def get_qfile_list(self):
        qfile_list = []
        for dirpath, dirnames, filenames in os.walk(self.data_path):
            for fn in filenames:
                if fn.endswith('.txt'):
                    qfile_list.append(os.path.join(dirpath, fn))

        print '[%s]' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        print '[%s]'% self.data_path
        print '\tqfiles size: %d'%len(qfile_list)
        return qfile_list

    def get_all_batch(self):
        data = []
        qid_rel_uid = {}
        for Local_feat, X1, X2, Y, curr_batch in self.get_batch():
            data.append([Local_feat, X1, X2, Y, curr_batch])
            for qid, uid, rel in curr_batch:
                if qid not in qid_rel_uid:
                    qid_rel_uid[qid] = {}
                if rel not in qid_rel_uid[qid]:
                    qid_rel_uid[qid][rel] = []
                qid_rel_uid[qid][rel].append(uid)

        print '[%s]' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        print '[ListGenerator] Read all batch done!'
        return data, qid_rel_uid

    def get_batch(self):
        qfile_idx = 0
        while True:
            Local_feat = np.zeros((self.batch_size, self.query_maxlen, self.doc_maxlen), dtype=np.int32)
            X1 = np.zeros((self.batch_size, self.query_maxlen), dtype=np.int32)
            X2 = np.zeros((self.batch_size, self.doc_maxlen), dtype=np.int32)
            Y = np.zeros((self.batch_size,), dtype=np.int32)
            Y[::2] = 1

            curr_batch = []
            for i in range(self.batch_size):
                line = self.data_handler.readline()
                if line == '':
                    if qfile_idx == len(self.qfile_list) - 1:
                        break
                    else:
                        qfile_idx += 1
                        self.data_handler.close()
                        self.data_handler = open(self.qfile_list[qfile_idx])
                        line = self.data_handler.readline()

                # qid, query, doc_id, doc, label = line.strip().split('\t')
                # qid = qid.strip()
                # doc_id = doc_id.strip()
                # label = float(label)
                
                attr = line.strip().split('\t')
                qid = attr[0].strip()
                query = attr[1].strip()
                doc_id = attr[2].strip()
                doc = attr[3].strip()
                label = float(attr[self.label_index])
                curr_batch.append([qid, doc_id, label])

                query = convert_term2id(query.strip().split(), self.word_dict)
                doc = convert_term2id(doc.strip().split(), self.word_dict)

                query_len = min(self.query_maxlen, len(query))
                doc_len   = min(self.doc_maxlen,   len(doc))

                # generate local feature matrix
                query_index = {}
                for query_i in range(query_len):
                    wid = query[query_i]
                    if wid in query_index:
                        query_index[wid].append(query_i)
                    else:
                        query_index[wid] = [query_i]
                for doc_i in range(doc_len):
                    wid = doc[doc_i]
                    if wid not in query_index:
                        continue
                    q_i_list = query_index[wid]
                    for q_i in q_i_list:
                        Local_feat[i, q_i, doc_i] = 1

                X1[i, :query_len] = query[:query_len]
                X2[i, :doc_len]   = doc[:doc_len]

            if len(curr_batch) < 1:
                break
            yield Local_feat, X1, X2, Y, curr_batch

    def get_batch_generator(self):
        for Local_feat, X1, X2, Y, curr_batch in self.data:
            yield ({'local_feat': Local_feat, 'query': X1, 'doc': X2}, Y, curr_batch)
