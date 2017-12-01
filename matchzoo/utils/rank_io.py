#! encoding: utf-8
#! author: pangliang

import json
import numpy as np
import re


import time

# def cal_hist(config):
#     hist_feats_all = {}
#     if 'hist_feats_file' in config:
#         for k in config:
#             if 'hist_feats_file' in k:
#                 hist_feats = read_features(config[k], config['hist_size'])
#                 hist_feats_all.update(hist_feats)
#         return hist_feats_all
#     # print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), ''
#     data1_maxlen = config['text1_maxlen']
#     data2_maxlen = config['text2_maxlen']
#     hist_size = config['hist_size']
#
#     embed = config['embed']
#     rel_file_cnt = 0
#     for key in config:
#         if 'relation_file' in key:
#             rel_file_cnt += 1
#     print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),'[%d rel files] Start calculating hist...'%rel_file_cnt,
#     cnt = 0
#     for key in config:
#         if 'relation_file' in key:
#             cnt += 1
#             print str(cnt)+'...',
#             rel_file = config[key]
#             rel = read_relation(filename=rel_file, verbose=False)
#             hist_feats = {}
#             for label, d1, d2 in rel:
#                 if d1 not in config['data1']:
#                     continue
#                 if d2 not in config['data2']:
#                     continue
#                 mhist = np.zeros((data1_maxlen, hist_size), dtype=np.float32)
#                 t1_rep = embed[config['data1'][d1][:data1_maxlen]]
#                 t2_rep = embed[config['data2'][d2][:data2_maxlen]]
#                 mm = t1_rep.dot(np.transpose(t2_rep))
#                 for (i, j), v in np.ndenumerate(mm):
#                     vid = int((v + 1.) / 2. * (hist_size - 1.))
#                     mhist[i][vid] += 1.
#                 mhist += 1.
#                 mhist = np.log10(mhist)
#                 hist_feats[(d1, d2)] = mhist
#
#             hist_feats_all.update(hist_feats)
#             output = open(rel_file.replace('.txt', '')+'_hist_%d.txt'%config['hist_size'], 'w')
#             for k, v in hist_feats.items():
#                 output.write('%s\t%s\t%s\n'%(k[0], k[1], ' '.join(map(str, np.reshape(v, [-1])))))
#             output.close()
#     print ''
#     # print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'cal_hist done!'
#     return hist_feats_all


def read_word_dict_zyk(word_dict_filepath):
    word_dict = {}
    with open(word_dict_filepath) as file:
        cnt = -1
        for line in file:
            cnt += 1
            if cnt == 0:
                continue
            attr = line.split(' ', 2)
            w = attr[0].lower().strip()
            if len(w) < 1:
                continue
            # id = cnt # int(attr[1].strip())
            if w not in word_dict:
                word_dict[w] = cnt
    print '[%s]' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '[%s]\n\tWord dict size: %d' % (word_dict_filepath, cnt+1)
    return word_dict, cnt+1


def read_embedding(filename):
    embed = {}
    word_dict = {}
    cnt = -1
    # max_cnt = max(word_ids.keys())
    for line in open(filename):
        cnt += 1
        if cnt == 0:
            continue
        # if cnt > 1000:
        #     break
        # if cnt in word_ids:
        attr = line.strip().split()
        if len(attr) != 51:
            continue
        term = attr[0].strip().lower()
        if len(term) < 1:
            continue
        if term not in word_dict:
            word_dict[term] = cnt
        # new_id = word_ids[cnt]
        embed[cnt] = map(float, attr[1:])

    print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '[%s]\n\tEmbedding size: %d' % (filename, len(embed))
    return embed, cnt+1, len(embed[1]), word_dict

# Read old version data
# def read_data_old_version(filename):
#     data = []
#     for idx, line in enumerate(open(filename)):
#         line = line.strip().split()
#         len1 = int(line[1])
#         len2 = int(line[2])
#         data.append([map(int, line[3:3+len1]), map(int, line[3+len1:])])
#         assert len2 == len(data[idx][1])
#     print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '[%s]\n\tInstance size: %d' % (filename, len(data))
#     return data

# Read Relation Data
# def read_relation(filename, verbose=True):
#     # print 'reading relation file', filename
#     data = []
#     # cnt = 0
#     for line in open(filename):
#         # cnt += 1
#         # print cnt
#         line = re.split('\t| ', line.strip())
#         data.append( (float(line[2]), line[0], line[1]) )
#     if verbose:
#         print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '[%s]\n\tInstance size: %d' % (filename, len(data))
#     return data

# Read varied-length features without id
# def read_features(filename, hist_size, verbose=True):
#     features = {}
#     for line in open(filename):
#         line = re.split('\t| ', line.strip())
#         d1 = line[0]
#         d2 = line[1]
#         features[(d1, d2)] = np.reshape(map(float, line[2:], [hist_size, -1]))
#     if verbose:
#         print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '[%s]\n\tFeature size: %d' % (filename, len(features))
#     return features

def read_idf(filename, word_dict):
    idfs = {}
    for line in open(filename):
        term, idf = line.split('\t')
        term = term.strip().lower()
        if len(term) < 1:
            continue
        if term not in word_dict:
            continue
        idf = float(idf)
        id = word_dict[term]
        if term not in idfs:
            idfs[id] = [idf]
    print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '[%s]\n\tIdf feat size: %d' % (filename, len(idfs))
    return idfs

def read_ngraph(filename):
    ngraph = {}
    cnt = -1
    for line in open(filename):
        cnt += 1
        if cnt > 2000:
            break
        term= line
        term = term.strip().lower()
        if len(term) < 1:
            continue
        ngraph[term] = cnt

    print '[%s]' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    print '[%s]\n\tngraphs feat size: %d' % (filename, len(ngraph))
    return ngraph, cnt+1


def convert_term2id(text, word_dict):
    new_text = []
    for term in text:
        term = term.strip().lower()
        if len(term) < 1:
            new_text.append(0)
            continue
        if term in word_dict:
            new_text.append(word_dict[term])
        else:
            new_text.append(0)
    assert len(new_text) == len(text)
    return new_text

# Convert Embedding Dict 2 numpy array
def convert_embed_2_numpy(name, embed_dict, max_size=-1, embed=None, normalize=False):
    feat_size = len(embed_dict[embed_dict.keys()[0]])
    if embed is None:
        embed = np.zeros( (max_size, feat_size), dtype = np.float32)
    for k in embed_dict:
        embed[k] = np.array(embed_dict[k], dtype=np.float32)
    if normalize:
        for i in range(len(embed)):
            embed[i] = embed[i]/np.linalg.norm(embed[i])
    print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'Generate numpy %s:'%name, embed.shape
    return embed

