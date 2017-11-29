import time
import numpy as np
from PairBasicGenerator import PairBasicGenerator

class Duet_PairGenerator(PairBasicGenerator):
    def __init__(self, config):
        super(Duet_PairGenerator, self).__init__(config=config)
        self.__name = 'Duet_PairGenerator'
        self.config = config
        # self.data1 = config['data1']
        # self.data2 = config['data2']
        self.data1_maxlen = config['text1_maxlen']
        self.data2_maxlen = config['text2_maxlen']
        self.fill_word = config['fill_word']
        # add for duet
        self.ngraphs = config['ngraphs']
        self.num_ngraphs = config['num_ngraphs']

        self.check_list.extend(['data1', 'data2', 'text1_maxlen', 'text2_maxlen', 'fill_word'])
        if config['use_iter']:
            self.batch_iter = self.get_batch_iter()
        if not self.check():
            raise TypeError('[Duet_PairGenerator] parameter check wrong.')
        print '[%s]' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '[Duet_PairGenerator] init done'

    # def load_ngraphs(self, filename):
    #     ngraphs = {}
    #     # max_ngraph_len = 0
    #     with open(filename) as f:
    #         for line in f:
    #             line = line.decode('utf-8', 'ignore')
    #             w, id = line.strip().split('\t')
    #             ngraphs[w] = int(id) - 1
    #             # max_ngraph_len = max(max_ngraph_len, len(w))
    #     return ngraphs, len(ngraphs)

    def get_batch_static(self):
        local_features = np.zeros((self.batch_size * 2, self.data1_maxlen, self.data2_maxlen), dtype=np.int32)
        ngraph_X1 = np.zeros((self.batch_size * 2, self.num_ngraphs, self.data1_maxlen), dtype=np.int32)
        ngraph_X2 = np.zeros((self.batch_size * 2, self.num_ngraphs, self.data2_maxlen), dtype=np.int32)
        ngraph_X1_len = np.zeros((self.batch_size * 2,), dtype=np.int32)
        ngraph_X2_len = np.zeros((self.batch_size * 2,), dtype=np.int32)
        Y = np.zeros((self.batch_size * 2,), dtype=np.int32)

        Y[::2] = 1
        for i in range(self.batch_size):
            d1, d2p, d2n = random.choice(self.pair_list)
            d1_len = min(self.data1_maxlen, len(self.data1[d1]))
            d2p_len = min(self.data2_maxlen, len(self.data2[d2p]))
            d2n_len = min(self.data2_maxlen, len(self.data2[d2n]))

            ngraph_X1_len[i * 2] = d1_len
            ngraph_X2_len[i * 2] = d2p_len
            ngraph_X1_len[i * 2 + 1] = d1_len
            ngraph_X2_len[i * 2 + 1] = d2n_len

            for d1_idx, d1_w in enumerate(self.data1[d1]):
                if d1_idx >= self.data1_maxlen:
                    break
                for d2p_idx, d2p_w in enumerate(self.data2[d2p]):
                    if d2p_idx >= self.data2_maxlen:
                        break
                    if d1_w == d2p_w:
                        local_features[i * 2, d1_idx, d2p_idx] = 1
                for d2n_idx, d2n_w in enumerate(self.data2[d2n]):
                    if d2n_idx >= self.data2_maxlen:
                        break
                    if d1_w == d2n_w:
                        local_features[i * 2 + 1, d1_idx, d2n_idx] = 1

            samples = [self.data1[d1], self.data2[d2p], self.data2[d2n]]

            for idx, data in enumerate(samples):
                ngraph_X = ngraph_X1 if idx == 0 else ngraph_X2
                max_words = self.data1_maxlen if idx == 0 else self.data2_maxlen
                for w_idx, word in enumerate(data[:min(len(data), max_words)]):
                    if word not in self.ngraphs:
                        continue
                    for nidx in self.ngraphs[word]:
                        if nidx >= self.num_ngraphs:
                            continue
                        if idx == 0:
                            ngraph_X[i * 2, nidx, w_idx] += 1
                            ngraph_X[i * 2 + 1, nidx, w_idx] += 1
                        else:
                            ngraph_X[i * 2 + idx - 1, nidx, w_idx] += 1

            # for idx, data in enumerate(samples):
            #     ngraph_X = ngraph_X1 if idx == 0 else ngraph_X2
            #     max_words = self.data1_maxlen if idx == 0 else self.data2_maxlen
            #     for w_idx, word in enumerate(data[:min(len(data), max_words)]):
            #         token = '#' + word + '#'
            #         token_len = len(token)
            #         for i in range(token_len):
            #             for j in range(i + 1, token_len + 1):
            #                 # if i + j > token_len:
            #                 #     continue
            #                 token_tmp = token[i:j]
            #                 # print('toke:\t', token_tmp)
            #                 ngraph_idx = self.ngraphs.get(token_tmp)
            #                 if ngraph_idx == None:
            #                     continue
            #                 if idx == 0:
            #                     ngraph_X[i * 2, ngraph_idx, w_idx] += 1
            #                     ngraph_X[i * 2 + 1, ngraph_idx, w_idx] += 1
            #                 else:
            #                     ngraph_X[i * 2 + idx - 1, ngraph_idx, w_idx] += 1

        return local_features, ngraph_X1, ngraph_X1_len, ngraph_X2, ngraph_X2_len, Y

    def get_batch_iter(self):
        while True:
            self.pair_list = self.pair_list_iter.next()
            for _ in range(self.config['batch_per_iter']):
                local_features = np.zeros((self.batch_size * 2, self.data1_maxlen, self.data2_maxlen), dtype=np.int32)
                ngraph_X1 = np.zeros((self.batch_size * 2, self.num_ngraphs, self.data1_maxlen), dtype=np.int32)
                ngraph_X2 = np.zeros((self.batch_size * 2, self.num_ngraphs, self.data2_maxlen), dtype=np.int32)
                ngraph_X1_len = np.zeros((self.batch_size * 2,), dtype=np.int32)
                ngraph_X2_len = np.zeros((self.batch_size * 2,), dtype=np.int32)
                Y = np.zeros((self.batch_size * 2,), dtype=np.int32)

                Y[::2] = 1
                for i in range(self.batch_size):
                    d1, d2p, d2n = random.choice(self.pair_list)
                    d1_len = min(self.data1_maxlen, len(self.data1[d1]))
                    d2p_len = min(self.data2_maxlen, len(self.data2[d2p]))
                    d2n_len = min(self.data2_maxlen, len(self.data2[d2n]))

                    ngraph_X1_len[i * 2] = d1_len
                    ngraph_X2_len[i * 2] = d2p_len
                    ngraph_X1_len[i * 2 + 1] = d1_len
                    ngraph_X2_len[i * 2 + 1] = d2n_len

                    for d1_idx, d1_w in enumerate(self.data1[d1]):
                        if d1_idx >= self.data1_maxlen:
                            break
                        for d2p_idx, d2p_w in enumerate(self.data2[d2p]):
                            if d2p_idx >= self.data2_maxlen:
                                break
                            if d1_w == d2p_w:
                                local_features[i * 2, d1_idx, d2p_idx] = 1
                        for d2n_idx, d2n_w in enumerate(self.data2[d2n]):
                            if d2n_idx >= self.data2_maxlen:
                                break
                            if d1_w == d2n_w:
                                local_features[i * 2 + 1, d1_idx, d2n_idx] = 1

                    samples = [self.data1[d1], self.data2[d2p], self.data2[d2n]]

                    for idx, data in enumerate(samples):
                        ngraph_X = ngraph_X1 if idx == 0 else ngraph_X2
                        max_words = self.data1_maxlen if idx == 0 else self.data2_maxlen
                        for w_idx, word in enumerate(data[:min(len(data), max_words)]):
                            if word not in self.ngraphs:
                                continue
                            for nidx in self.ngraphs[word]:
                                if nidx >= self.num_ngraphs:
                                    continue
                                if idx == 0:
                                    ngraph_X[i * 2, nidx, w_idx] += 1
                                    ngraph_X[i * 2 + 1, nidx, w_idx] += 1
                                else:
                                    ngraph_X[i * 2 + idx - 1, nidx, w_idx] += 1
                yield local_features, ngraph_X1, ngraph_X1_len, ngraph_X2, ngraph_X2_len, Y
                    # for idx, data in enumerate(samples):
                    #     ngraph_X = ngraph_X1 if idx == 0 else ngraph_X2
                    #     max_words = self.data1_maxlen if idx == 0 else self.data2_maxlen
                    #     for w_idx, word in enumerate(data[:min(len(data), max_words)]):
                    #         token = '#' + word + '#'
                    #         token_len = len(token)
                    #         for i in range(token_len):
                    #             for j in range(i+1, token_len + 1):
                    #                 # if i + j > token_len:
                    #                 #     continue
                    #                 token_tmp = token[i:j]
                    #                 # print('toke:\t', token_tmp)
                    #                 ngraph_idx = self.ngraphs.get(token_tmp)
                    #                 if ngraph_idx == None:
                    #                     continue
                    #                 if idx == 0:
                    #                     ngraph_X[i * 2, ngraph_idx, w_idx] += 1
                    #                     ngraph_X[i * 2 + 1, ngraph_idx, w_idx] += 1
                    #                 else:
                    #                     ngraph_X[i * 2 + idx - 1, ngraph_idx, w_idx] += 1
                # print local_features.shape
                # print ngraph_X1.shape
                # print ngraph_X2.shape


    def get_batch_generator(self):
        while True:
            local_features, X1, X1_len, X2, X2_len, Y = self.get_batch()
            # if self.config['use_dpool']:
            #     yield ({'local_features': local_features, 'query': X1, 'query_len': X1_len, 'doc': X2,'doc_len': X2_len,
            #             'dpool_index': DynamicMaxPooling.dynamic_pooling_index(X1_len, X2_len,
            #                                                                    self.config['text1_maxlen'],
            #                                                                    self.config['text2_maxlen'])}, Y)
            # else:
            yield ({'local_feats': local_features, 'query': X1, 'query_len': X1_len, 'doc': X2,'doc_len': X2_len}, Y)

