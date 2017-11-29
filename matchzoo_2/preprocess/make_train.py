import os

high_label = 3.0
rel_gap = 1.0
output = open('real_train_set.txt', 'w')
for dirpath, dirnames, filenames in os.walk('tmp'):
    for fn in filenames:
        if not fn.endswith('.txt'):
            continue
        qid = fn.replace('.txt', '')
        query = ''
        file = open(os.path.join(dirpath, fn))
        tmp_uid_label = {}
        uid_doc = {}
        for line in file:
            _qid, _query, uid, doc, label = line.split('\t')
            query = _query.strip()
            uid = uid.strip()
            label = float(label)
            doc = doc.strip()
            uid_doc[uid] = doc
            if label not in tmp_uid_label:
                tmp_uid_label[label] = []
            tmp_uid_label[label].append(uid)
        file.close()
        for hl in tmp_uid_label:
            for ll in tmp_uid_label:
                if hl <= ll:
                    continue
                if hl < high_label:
                    continue
                if hl - ll <= rel_gap:
                    continue
                for d_p in tmp_uid_label[hl]:
                    for d_n in tmp_uid_label[ll]:
                        output.write(qid+'\t'+query+'\t')
                        output.write(d_p + '\t' + uid_doc[d_p] + '\t' + str(hl) + '\t')
                        output.write(d_n + '\t' + uid_doc[d_n] + '\t' + str(ll) + '\n')


output.close()

