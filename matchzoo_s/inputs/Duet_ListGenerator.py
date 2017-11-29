# # -*- coding: utf-8 -*-
#
# import sys
# import random
# import numpy as np
# import time
#
# from utils.rank_io import *
# from layers import DynamicMaxPooling
#
# import scipy.sparse as sp
#
# EPS = 1e-20
#
#
# class Duet_ListGenerator(ListBasicGenerator):
#     def __init__(self, config={}):
#         super(Duet_ListGenerator, self).__init__(config=config)
#         self.__name = 'Duet_ListGenerator'
#         # self.data1 = config['data1']
#         # self.data2 = config['data2']
#         self.data1_maxlen = config['text1_maxlen']
#         self.data2_maxlen = config['text2_maxlen']
#         # self.fill_word = config['fill_word']
#         # add for duet
#         self.ngraphs = config['ngraphs']
#         self.num_ngraphs = config['num_ngraphs']
#         self.new_list_list = self.list_list_transfer()
#         self.num_new_list = len(self.new_list_list)
#         self.check_list.extend(['data1', 'data2', 'text1_maxlen', 'text2_maxlen'])
#         if not self.check():
#             raise TypeError('[Duet_ListGenerator] parameter check wrong.')
#         print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '[Duet_ListGenerator] init done'
#
#     def list_list_transfer(self):
#         list_list = self.list_list
#         new_list_list = []
#         for d1, d2_list in list_list:
#             for l, d2 in d2_list:
#                 new_list_list.append([d1, d2, l])
#         return new_list_list
#
#     def get_batch(self):
#         while self.point < self.num_new_list:
#             currbatch = []
#             if self.point + self.batch_list <= self.num_new_list:
#                 currbatch = self.new_list_list[self.point: self.point+self.batch_list]
#                 self.point += self.batch_list
#             else:
#                 currbatch = self.new_list_list[self.point:]
#                 self.point = self.num_new_list
#
#             new_currbatch = {}
#             for d1, d2, l in currbatch:
#                 if d1 not in new_currbatch:
#                     new_currbatch[d1] = []
#                 new_currbatch[d1].append([l, d2])
#             currbatch = new_currbatch.items()
#
#             bsize = sum([len(pt[1]) for pt in currbatch])
#             ID_pairs = []
#             list_count = [0]
#
#             local_features = np.zeros((bsize, self.data1_maxlen, self.data2_maxlen), dtype=np.int32)
#             ngraph_X1 = np.zeros((bsize, self.num_ngraphs, self.data1_maxlen), dtype=np.int32)
#             ngraph_X2 = np.zeros((bsize, self.num_ngraphs, self.data2_maxlen), dtype=np.int32)
#
#             # X1 = np.zeros((bsize, self.data1_maxlen), dtype=np.int32)
#             X1_len = np.zeros((bsize,), dtype=np.int32)
#             # X2 = np.zeros((bsize, self.data2_maxlen), dtype=np.int32)
#             X2_len = np.zeros((bsize,), dtype=np.int32)
#             Y = np.zeros((bsize,), dtype= np.int32)
#
#             j = 0
#             for pt in currbatch:
#                 d1, d2_list = pt[0], pt[1]
#                 list_count.append(list_count[-1] + len(d2_list))
#                 d1_len = min(self.data1_maxlen, len(self.data1[d1]))
#                 for l, d2 in d2_list:
#                     d2_len = min(self.data2_maxlen, len(self.data2[d2]))
#                     X1_len[j] = d1_len
#                     X2_len[j] = d2_len
#                     for d1_idx, d1_w in enumerate(self.data1[d1]):
#                         if d1_idx >= self.data1_maxlen:
#                             break
#                         for d2_idx, d2_w in enumerate(self.data2[d2]):
#                             if d2_idx >= self.data2_maxlen:
#                                 break
#                             if d1_w == d2_w:
#                                 local_features[j, d1_idx, d2_idx] = 1
#                     samples = [self.data1[d1], self.data2[d2]]
#                     for idx, data in enumerate(samples):
#                         ngraph_X = ngraph_X1 if idx == 0 else ngraph_X2
#                         max_words = self.data1_maxlen if idx == 0 else self.data2_maxlen
#                         for w_idx, word in enumerate(data[:min(len(data), max_words)]):
#                             if word not in self.ngraphs:
#                                 continue
#                             for nidx in self.ngraphs[word]:
#                                 if nidx >= self.num_ngraphs:
#                                     continue
#                                 # if idx == 0:
#                                 ngraph_X[j, nidx, w_idx] += 1
#                                 # else:
#                                 #     ngraph_X[j, nidx, w_idx] += 1
#
#                     ID_pairs.append((d1, d2))
#                     Y[j] = l
#                     j += 1
#             self.currbatch = currbatch
#             yield local_features, ngraph_X1, X1_len, ngraph_X2, X2_len, Y, ID_pairs, list_count
#
#     def get_batch_generator(self):
#         for local_features, X1, X1_len, X2, X2_len, Y, ID_pairs, list_counts in self.get_batch():
#             # if self.config['use_dpool']:
#             #     yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len, 'dpool_index': DynamicMaxPooling.dynamic_pooling_index(X1_len, X2_len, self.config['text1_maxlen'], self.config['text2_maxlen']), 'ID': ID_pairs, 'list_counts': list_counts}, Y)
#             # else:
#             yield ({'local_feats': local_features, 'query': X1, 'query_len': X1_len, 'doc': X2,'doc_len': X2_len, 'ID': ID_pairs, 'list_counts': list_counts}, Y)
#
#     # def get_all_data(self):
#     #     x1_ls, x1_len_ls, x2_ls, x2_len_ls, y_ls, list_count_ls = [], [], [], [], [], []
#     #     while self.point < self.num_list:
#     #         currbatch = []
#     #         if self.point + self.batch_list <= self.num_list:
#     #             currbatch = self.list_list[self.point: self.point+self.batch_list]
#     #             self.point += self.batch_list
#     #         else:
#     #             currbatch = self.list_list[self.point:]
#     #             self.point = self.num_list
#     #
#     #         bsize = sum([len(pt[1]) for pt in currbatch])
#     #         list_count = [0]
#     #         X1 = np.zeros((bsize, self.data1_maxlen), dtype=np.int32)
#     #         X1_len = np.zeros((bsize,), dtype=np.int32)
#     #         X2 = np.zeros((bsize, self.data2_maxlen), dtype=np.int32)
#     #         X2_len = np.zeros((bsize,), dtype=np.int32)
#     #         Y = np.zeros((bsize,), dtype= np.int32)
#     #         X1[:] = self.fill_word
#     #         X2[:] = self.fill_word
#     #         j = 0
#     #         for pt in currbatch:
#     #             d1, d2_list = pt[0], pt[1]
#     #             list_count.append(list_count[-1] + len(d2_list))
#     #             d1_len = min(self.data1_maxlen, len(self.data1[d1]))
#     #             for l, d2 in d2_list:
#     #                 d2_len = min(self.data2_maxlen, len(self.data2[d2]))
#     #                 X1[j, :d1_len], X1_len[j] = self.data1[d1][:d1_len], d1_len
#     #                 X2[j, :d2_len], X2_len[j] = self.data2[d2][:d2_len], d2_len
#     #                 Y[j] = l
#     #                 j += 1
#     #         x1_ls.append(X1)
#     #         x1_len_ls.append(X1_len)
#     #         x2_ls.append(X2)
#     #         x2_len_ls.append(X2_len)
#     #         y_ls.append(Y)
#     #         list_count_ls.append(list_count)
#     #     return x1_ls, x1_len_ls, x2_ls, x2_len_ls, y_ls, list_count_ls
