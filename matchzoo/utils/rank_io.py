#! encoding: utf-8
#! author: pangliang

import json
import numpy as np
import re

# Read Word Dict and Inverse Word Dict
def read_word_dict(filename):
    word_dict = {}
    iword_dict = {}
    for line in open(filename):
        line = line.strip().split()
        word_dict[int(line[1])] = line[0]
        iword_dict[line[0]] = int(line[1])
    print '[%s]\n\tWord dict size: %d' % (filename, len(word_dict))
    return word_dict, iword_dict


# Read Embedding File
# def read_embedding(filename, word_ids):
#     embed = {}
#     len_word_ids = len(word_ids)
#     word_ids = sorted(word_ids.items(), key=lambda x:x[0])
#     cnt = -1
#     ids_idx = 0
#     for line in open(filename):
#         if cnt == -1:
#             cnt += 1
#             continue
#         line = line.strip().split()
#         if ids_idx >= len_word_ids:
#             break
#         if cnt == word_ids[ids_idx][0]:
#             embed[word_ids[ids_idx][1]] = map(float, line[1:])
#             # embed[cnt] = np.
#             ids_idx += 1
#         cnt += 1
#     print '[%s]\n\tEmbedding size: %d' % (filename, len(embed))
#     return embed

def read_embedding(filename, word_ids, normalize=False):
    embed = {}
    cnt = -2
    max_cnt = max(word_ids.keys())
    for line in open(filename):
        cnt += 1
        if cnt == -1:
            continue
        if cnt > max_cnt:
            break
        if cnt in word_ids:
            line = line.strip().split()
            new_id = word_ids[cnt]
            embed[new_id] = map(float, line[1:])
            if normalize:
                x = np.array(embed[new_id], dtype=np.float32)
                embed[new_id] = (x / np.linalg.norm(x)).tolist()

    print '[%s]\n\tEmbedding size: %d' % (filename, len(embed))
    return embed

# Read old version data
def read_data_old_version(filename):
    data = []
    for idx, line in enumerate(open(filename)):
        line = line.strip().split()
        len1 = int(line[1])
        len2 = int(line[2])
        data.append([map(int, line[3:3+len1]), map(int, line[3+len1:])])
        assert len2 == len(data[idx][1])
    print '[%s]\n\tInstance size: %d' % (filename, len(data))
    return data

# Read Relation Data
def read_relation(filename, verbose=True):
    # print 'reading relation file', filename
    data = []
    # cnt = 0
    for line in open(filename):
        # cnt += 1
        # print cnt
        line = re.split('\t| ', line.strip())
        data.append( (float(line[2]), line[0], line[1]) )
    if verbose:
        print '[%s]\n\tInstance size: %s' % (filename, len(data))
    return data

# Read varied-length features without id
def read_features(filename, verbose=True):
    features = []
    for line in open(filename):
        line = line.strip().split()
        features.append(map(float, line))
    if verbose:
        print '[%s]\n\tFeature size: %s' % (filename, len(features))
    return features

# def read_idf(filename, word_dict = None):
#     data = {}
#     data[-1] = []
#     for line in open(filename):
#         line = re.split('\t', line[:-1])
#         term = line[0]
#         idf = float(line[1])
#         if word_dict == None:
#             print "error!"
#             exit(0)
#         if term not in word_dict or word_dict[term] == -1:
#             data[-1].append(idf)
#         else:
#             data[word_dict[term]] = idf
#     data[-1] = sum(data[-1])/len(data[-1])
#     # print '[%s]\n\tidf feat size: %s' % (filename, len(data))
#     return data

def read_idf(filename, word_dict = None):
    data = {}
    data[-1] = 0
    for line in open(filename):
        line = re.split('\t', line[:-1])
        term = line[0].lower().strip()
        idf = float(line[1])
        if word_dict == None:
            print "error!"
            exit(0)
        if term not in word_dict or word_dict[term] == -1:
            continue
            # data[-1].append(idf)
        # else:
        data[word_dict[term]] = idf
    data[-1] = sorted(data.values())[len(data)/2] # median
    # print '[%s]\n\tidf feat size: %s' % (filename, len(data))
    return data

# Read Data Dict
def read_data(filename, word_dict = None):
    data = {}
    data_word = []
    for line in open(filename):
        line = re.split('\t| ', line.strip())
        tid = line[0]
        # data = line[1].split()
        if word_dict == None:
            data[tid] = map(int, line[1:])
        else:
            data[tid] = []
            for w in line[1:]:
                w = w.lower()
                if w not in word_dict or word_dict[w] == -1:
                    word_dict[w] = -1
                    data[tid].append(-1)
                else:
                    data_word.append(word_dict[w])
                    data[tid].append(word_dict[w])
    print '[%s]\n\tData size: %s' % (filename, len(data))
    return data, sorted(list(set(data_word)))

# Convert Embedding Dict 2 numpy array
def convert_embed_2_numpy(embed_dict, max_size=0, embed=None):
    feat_size = len(embed_dict[embed_dict.keys()[0]])
    if embed is None:
        embed = np.zeros( (max_size, feat_size), dtype = np.float32 )
    for k in embed_dict:
        embed[k] = np.array(embed_dict[k], dtype=np.float32)
    print 'Generate numpy embed:', embed.shape
    return embed

