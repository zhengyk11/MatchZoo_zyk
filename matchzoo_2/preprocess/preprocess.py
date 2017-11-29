import json
import os
config = """
{
    "qrel":"../../data/fulltext/real_train/real_train_qrel/real_train_qrel.txt",
    "doc":"../../data/fulltext/real_train/real_train_fulltext_seg_filter.txt",
    "query":"../../data/fulltext/real_train/real_train_qid_seg.txt",
    "output_dir": "../../runtime_data/fulltext/real_train"
}
"""

uid_qid_label = {}
qid_query = {}
qids = {}
uids = {}
config = json.loads(config)

if not os._exists(config['output_dir']):
    os.mkdir(config['output_dir'])

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
                    with open(config['output_dir']+'/'+qid+'.txt', 'w') as output:
                        output.write(qid + '\t' + qid_query[qid] + '\t' + uid + '\t' + doc + '\t' + uid_qid_label[uid][qid] + '\n')
                    qids[qid] = 0
                else:
                    with open(config['output_dir']+'/'+qid+'.txt', 'a') as output:
                        output.write(qid + '\t' + qid_query[qid] + '\t' + uid + '\t' + doc + '\t' +uid_qid_label[uid][qid]+ '\n')