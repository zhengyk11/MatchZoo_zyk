import random
import time

from matchzoo import convert_term2id


class PairBasicGenerator(object):
    def __init__(self, config):
        self.__name = 'PairBasicGenerator'
        self.config = config
        self.data1 = config['data1']
        self.data2 = config['data2']
        self.word_dict = config['word_dict']
        if 'rel_gap' in config:
            self.rel_gap = config['rel_gap']
        else:
            self.rel_gap = 0.
        if 'high_label' in config:
            self.high_label = config['high_label']
        else:
            self.high_label = 0.
        self.rel = []
        for key in config:
            if 'relation_file' in key:
                rel_file = config[key]
                self.rel += read_relation(filename=rel_file)
        self.rel = list(set(self.rel))
        # rel_file = config['relation_file']
        # self.rel = read_relation(filename=rel_file)
        self.batch_size = config['batch_size']
        self.check_list = ['relation_file', 'batch_size']
        self.point = 0
        if config['use_iter']:
            self.pair_list_iter = self.make_pair_iter(self.rel)
            self.pair_list = []
        else:
            self.pair_list = self.make_pair_static(self.rel)
            self.pair_list_iter = None

    def check(self):
        for e in self.check_list:
            if e not in self.config:
                print '[%s]' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '[%s] Error %s not in config' % (
                self.__name, e)
                return False
        return True

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
                    doc = convert_term2id(doc.strip().split(), self.word_dict)
                    label = float(label)
                    qid_query[qid] = query
                    uid_doc[uid] = doc
                    if qid not in qid_label_uid:
                        qid_label_uid[qid] = {}
                    if label not in qid_label_uid[qid]:
                        qid_label_uid[qid][label] = []
                    qid_label_uid[qid][label].append(uid)

        return qid_query, uid_doc, qid_label_uid

    def make_pair_static(self, rel):
        rel_set = {}
        pair_list = []
        for label, d1, d2 in rel:
            if d1 not in self.data1:
                continue
            if d2 not in self.data2:
                continue
            if d1 not in rel_set:
                rel_set[d1] = {}
            if label not in rel_set[d1]:
                rel_set[d1][label] = []
            rel_set[d1][label].append(d2)

        for d1 in rel_set:
            label_list = sorted(rel_set[d1].keys(), reverse=True)
            for hidx, high_label in enumerate(label_list[:-1]):
                for low_label in label_list[hidx + 1:]:
                    if high_label < self.high_label:
                        continue
                    if high_label - low_label <= self.rel_gap:
                        continue
                    for high_d2 in rel_set[d1][high_label]:
                        for low_d2 in rel_set[d1][low_label]:
                            pair_list.append((d1, high_d2, low_d2))
        print '[%s]' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'Pair Instance Count:', len(pair_list)
        return pair_list

    def make_pair_iter(self, rel):
        rel_set = {}
        pair_list = []
        for label, d1, d2 in rel:
            if d1 not in self.data1:
                continue
            if d2 not in self.data2:
                continue
            if d1 not in rel_set:
                rel_set[d1] = {}
            if label not in rel_set[d1]:
                rel_set[d1][label] = []
            rel_set[d1][label].append(d2)

        while True:
            pair_list = []
            rel_set_sample = random.sample(rel_set.keys(), self.config['query_per_iter'])

            for d1 in rel_set_sample:
                label_list = sorted(rel_set[d1].keys(), reverse=True)
                for hidx, high_label in enumerate(label_list[:-1]):
                    for low_label in label_list[hidx + 1:]:
                        if high_label < self.high_label:
                            continue
                        if high_label - low_label <= self.rel_gap:
                            continue
                        for high_d2 in rel_set[d1][high_label]:
                            for low_d2 in rel_set[d1][low_label]:
                                pair_list.append((d1, high_d2, low_d2))
            if len(pair_list) == 0:
                continue
            print '[%s]' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'Pair Instance Count:', len(pair_list)
            yield pair_list

    def get_batch_static(self):
        pass

    def get_batch_iter(self):
        pass

    def get_batch(self):
        if self.config['use_iter']:
            return self.batch_iter.next()
        else:
            return self.get_batch_static()

    def get_batch_generator(self):
        pass

    @property
    def num_pairs(self):
        return len(self.pair_list)

    def reset(self):
        self.point = 0