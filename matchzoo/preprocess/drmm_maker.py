# suitable for drmm

import multiprocessing
import time
import json
import os
import sys
sys.path.append('../')

from utils import read_embedding, convert_term2id, convert_embed_2_numpy
import numpy as np


def cal_hist(query_embed, doc_embed, config):
    hist = np.zeros([config['query_maxlen'], config['hist_size']], dtype=np.int32)
    hist[:] = 1
    mm = np.zeros([config['query_maxlen'], config['doc_maxlen']], dtype=np.float32)
    for i in range(config['query_maxlen']):
        for j in range(config['doc_maxlen']):
            mm[i][j] = np.dot(query_embed[i], doc_embed[j])
    for (i, j), v in np.ndenumerate(mm):
        vid = int((v + 1.) / 2. * (config['hist_size'] - 1.))
        hist[i][vid] += 1.

    return hist

def main():
    """
    config = 
    {
        "query_maxlen": 10,
        "doc_maxlen": 1000,
        "hist_size": 200,
        "embed_path": "../../data/support/embed_query_100w_50.txt",
        "input": "../../runtime_data/fulltext/bm25_train",
        "output": "../../runtime_data/fulltext/bm25_train_drmm"
    }
	"""
    config = """
    {
        "query_maxlen": 10,
        "doc_maxlen": 1000,
        "hist_size": 200,
        "embed_path": "../../../../K-NRM/data/embed_query_100w_50.txt",
        "input": "../../../../K-NRM/data/real_valid_HL_solrex",
        "output": "../../../../K-NRM/data/real_valid_HL_solrex_drmm"
    }
	"""
    #typenum = int(sys.argv[0])
    process_total = 40
    cpu_num = 2
    config = json.loads(config)

    embed_dict, vocab_size, embed_size, word_dict, idf_dict = read_embedding(config['embed_path'])
    embed = np.float32(np.random.uniform(-9, 9, [vocab_size, embed_size]))
    embed_dict = convert_embed_2_numpy('embed', embed_dict=embed_dict, embed=embed, normalize=True)

    if not os.path.exists(config['output']):
        os.mkdir(config['output'])
    pool = multiprocessing.Pool(processes=cpu_num)
    for i in range(0,process_total):
        pool.apply_async(getfile,(config,process_total,i,embed_dict,embed_size,word_dict))
    pool.close()
    pool.join()

def getfile(config, process_total, typenum, embed_dict, embed_size, word_dict):
    for dirpath, dirnames, filenames in os.walk(config['input']):
        for fn in filenames:
            if not fn.endswith('.txt'):
                continue
            if (int(fn.strip().split('.')[0]) % process_total != typenum):
                continue
            print os.path.join(dirpath, fn)
            with open(os.path.join(dirpath, fn)) as file:
                qid, query, uid, doc, label = file.readline().split('\t')
                output = open(os.path.join(config['output'], qid + '.txt'), 'w')
                query = query.strip().split()[:min(len(query), config['query_maxlen'])]
                query = convert_term2id(query, word_dict)
                query_embed = np.zeros([config['query_maxlen'], embed_size], dtype=np.float32)
                for i, wid in enumerate(query):
                    query_embed[i] = embed_dict[wid]
                file.seek(0)
                for line in file:
                    qid, query, uid, doc, label = line.split('\t')
                    uid = uid.strip()
                    qid = qid.strip()
                    query = query.strip()
                    label = label.strip()
                    doc = doc.strip().split()[:min(len(doc), config['doc_maxlen'])]
                    doc = convert_term2id(doc, word_dict)
                    doc_embed = np.zeros([config['doc_maxlen'], embed_size], dtype=np.float32)
                    for i, wid in enumerate(doc):
                        doc_embed[i] = embed_dict[wid]
                    hist = cal_hist(query_embed, doc_embed, config)
                    hist = ' '.join(map(str, np.reshape(hist, [-1])))
                    output.write(qid + '\t' + query + '\t' + uid + '\t' + hist + '\t' + label + '\n')

                output.close()




if __name__ == '__main__':
    main()

