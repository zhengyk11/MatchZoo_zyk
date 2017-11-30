# suitable for dssm & cdssm & duet & arci & arcii
import json
import os

# config = """
# {
#     "qrel_1":"../../data/fulltext/real_train/real_train_qrel/real_train_qrel_HL.txt",
#     "qrel_2": "../../data/fulltext/real_train/real_train_qrel/real_train_qrel_solrex.txt",
#     "doc_1": "../../data/fulltext/real_train/real_train_fulltext_seg_filter.txt",
#     "doc_2":"../../data/fulltext/real_train/real_train_fulltext_seg_filter_solrex.txt",
#     "query":"../../data/fulltext/real_train/real_train_qid_seg.txt",
#     "output": "../../runtime_data/fulltext/real_train_HL_solrex"
# }
# """

config = """
{
    "qrel_1": "../../data/fulltext/real_valid_test/real_valid_qrel/real_valid_qrel_HL.txt",
    "qrel_2": "../../data/fulltext/real_valid_test/real_valid_qrel/real_valid_qrel_solrex.txt",
    "doc_1":  "../../data/fulltext/real_valid_test/real_valid_test_fulltext_seg_filter.txt",
    "doc_2":  "../../data/fulltext/real_valid_test/real_valid_test_fulltext_seg_filter_solrex.txt",
    "query":  "../../data/fulltext/real_valid_test/real_valid_test_qid_seg.txt",
    "output": "../../runtime_data/fulltext/real_test_HL_solrex"
}
"""

# config = """
# {
#     "qrel":"../../data/fulltext/real_train/real_train_qrel/real_train_qrel.txt",
#     "doc":"../../data/fulltext/real_train/real_train_fulltext_seg_filter.txt",
#     "query":"../../data/fulltext/real_train/real_train_qid_seg.txt",
#     "output": "../../runtime_data/fulltext/real_train"
# }
# """

uid_qid_label = {}
qid_query = {}
qids = {}
uids = {}
config = json.loads(config)

if not os.path.exists(config['output']):
    os.mkdir(config['output'])

for k in config:
    if k.startswith('qrel'):
        for line in open(config[k]):
            qid, uid, label = line.split('\t')
            qid = qid.strip()
            uid = uid.strip()
            label = label.strip()
            if uid not in uid_qid_label:
                uid_qid_label[uid] = {}
            uid_qid_label[uid][qid] = label

for k in config:
    if k.startswith('query'):
        for line in open(config[k]):
            qid, query = line.split('\t')
            query = query.strip()# .split()
            qid = qid.strip()
            qid_query[qid] = query

for k in config:
    if k.startswith('doc'):
        for line in open(config[k]):
            uid, doc = line.split('\t')
            if uid in uids:
                continue
            uids[uid] = 0
            doc = doc.strip() # .split()
            uid = uid.strip()
            if uid not in uid_qid_label:
                continue
            for qid in uid_qid_label[uid]:
                if qid not in qids:
                    with open(config['output']+'/'+qid+'.txt', 'w') as output:
                        output.write(qid + '\t' + qid_query[qid] + '\t' + uid + '\t' + doc + '\t' + uid_qid_label[uid][qid] + '\n')
                    qids[qid] = 0
                else:
                    with open(config['output']+'/'+qid+'.txt', 'a') as output:
                        output.write(qid + '\t' + qid_query[qid] + '\t' + uid + '\t' + doc + '\t' +uid_qid_label[uid][qid]+ '\n')