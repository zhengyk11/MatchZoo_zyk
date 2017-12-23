import os
import time
import numpy as np
# from ListBasicGenerator import ListBasicGenerator
from utils import convert_term2id


class DRMM_ListGenerator(): # ListBasicGenerator):
    def __init__(self, config):
        # super(DRMM_ListGenerator, self).__init__(config=config)
        self.__name = 'DRMM_ListGenerator'

        self.query_maxlen = config['query_maxlen']
        self.hist_size = config['hist_size']
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

        self.log_option = False
        if 'log_opiton' in config:
            self.log_option = config['log_opiton']

        self.qfile_list = self.get_qfile_list()
        assert len(self.qfile_list) > 0
        self.data_handler = open(self.qfile_list[0])# self.get_data_handler()

        self.data, self.qid_rel_uid = self.get_all_batch()
        self.all_pairs = self.make_pairs()
        self.data_handler.close()

        print '[%s]' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        print '[DRMM_ListGenerator] init done'

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
        for X1, X2, Y, curr_batch in self.get_batch():
            data.append([X1, X2, Y, curr_batch])
            for qid, uid, rel in curr_batch:
                if qid not in qid_rel_uid:
                    qid_rel_uid[qid] = {}
                if rel not in qid_rel_uid[qid]:
                    qid_rel_uid[qid][rel] = []
                qid_rel_uid[qid][rel].append(uid)

        print '[%s]' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        print '[DSSM_ListGenerator] Read all batch done!'
        return data, qid_rel_uid

    def get_batch(self):
        qfile_idx = 0
        while True:
            X1 = np.zeros((self.batch_size, self.query_maxlen), dtype=np.int32)
            X2 = np.zeros((self.batch_size, self.query_maxlen, self.hist_size), dtype=np.float32)
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
                doc    = map(float, doc.split())

                doc_hist = np.reshape(doc, (self.query_maxlen, self.hist_size))

                if self.log_option:
                    doc_hist = np.log10(doc_hist)

                query_len = min(self.query_maxlen, len(query))

                X1[i, :query_len] = query[:query_len]
                X2[i] = doc_hist

            if len(curr_batch) < 1:
                break
            yield X1, X2, Y, curr_batch

    def get_batch_generator(self):
        for X1, X2, Y, curr_batch in self.data:
            yield ({'query': X1, 'doc': X2}, Y, curr_batch)
