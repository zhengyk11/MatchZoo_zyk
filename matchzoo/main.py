# -*- coding: utf8 -*-
import os
import sys
import time
import json
import argparse
import tensorflow as tf
from collections import OrderedDict
from keras.models import Model
import keras.backend.tensorflow_backend as KTF
# from keras.utils import multi_gpu_model


from utils import *
import inputs
import metrics
from losses import *


def train(config):
    print(json.dumps(config, indent=2))

    # read basic config
    global_conf = config["global"]
    optimizer = global_conf['optimizer']
    weights_file = global_conf['weights_file']
    num_batch = global_conf['num_batch']

    # read input config
    input_conf = config['inputs']
    share_input_conf = input_conf['share']

    # collect embedding
    assert 'embed_path' in share_input_conf
    embed_dict, vocab_size, embed_size, word_dict, idf_dict = read_embedding(share_input_conf['embed_path'])
    share_input_conf['word_dict'] = word_dict
    # share_input_conf['feat_size'] = vocab_size
    share_input_conf['vocab_size'] = vocab_size
    share_input_conf['embed_size'] = embed_size
    embed = np.float32(np.random.uniform(-9, 9, [vocab_size, embed_size]))
    embed_normalize = False
    if 'drmm' in config['model']['model_py'].lower():
        embed_normalize = True
    share_input_conf['embed'] = convert_embed_2_numpy('embed', embed_dict=embed_dict, embed=embed, normalize=embed_normalize)
    idf = np.float32(np.random.uniform(4, 9, [vocab_size, 1]))
    share_input_conf['idf_feat'] = convert_embed_2_numpy('idf', embed_dict=idf_dict, embed=idf, normalize=False)
    print '[%s]' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '[Embedding] Embedding Load Done.'

    # list all input tags and construct tags config
    input_train_conf = OrderedDict()
    input_eval_conf = OrderedDict()
    for tag in input_conf.keys():
        if 'phase' not in input_conf[tag]:
            continue
        if input_conf[tag]['phase'] == 'TRAIN':
            input_train_conf[tag] = {}
            input_train_conf[tag].update(share_input_conf)
            input_train_conf[tag].update(input_conf[tag])
        elif input_conf[tag]['phase'] == 'EVAL':
            input_eval_conf[tag] = {}
            input_eval_conf[tag].update(share_input_conf)
            input_eval_conf[tag].update(input_conf[tag])
    print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    print '[Input] Process Input Tags. %s in TRAIN, %s in EVAL.' % (input_train_conf.keys(), input_eval_conf.keys())

    # initial data generator
    train_gen = OrderedDict()
    eval_gen = OrderedDict()

    for tag, conf in input_train_conf.items():
        generator = inputs.get(conf['input_type'])
        train_gen[tag] = generator(config=conf)

    for tag, conf in input_eval_conf.items():
        generator = inputs.get(conf['input_type'])
        eval_gen[tag] = generator(config=conf)

    ######### Load Model #########
    _model = load_model(config)
    # model = multi_gpu_model(_model, gpus=2)
    model = _model
    loss = []
    for lobj in config['losses']:
        loss.append(rank_losses.get(lobj))
    eval_metrics = OrderedDict()
    for mobj in config['metrics']:
        mobj = mobj.lower()
        if '@' in mobj:
            mt_key, mt_val = mobj.split('@', 1)
            eval_metrics[mobj] = metrics.get(mt_key)(int(mt_val))
        else:
            eval_metrics[mobj] = metrics.get(mobj)
    model.compile(optimizer=optimizer, loss=loss)
    print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '[Model] Model Compile Done.\n'

    print '### Model Info ###'
    model.summary()
    print 'Total number of parameters:', model.count_params()
    print '### Model Info ###'

    for i_e in range(global_conf['num_epochs']):
        model.save_weights(weights_file)
        for tag, generator in train_gen.items():
            genfun = generator.get_batch_generator()
            num_batch_cnt = 0
            for input_data, y_true in genfun:
                num_batch_cnt += 1
                info = model.fit(x=input_data, y=y_true, epochs=1, verbose=0)
                print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                print '[Train] @ iter: %d,' % (i_e*num_batch+num_batch_cnt),
                print 'train_%s: %.5f' %(config['losses'][0], info.history['loss'][0])
                if num_batch_cnt == num_batch:
                    break

        for tag, generator in eval_gen.items():
            output_dir = config['net_name'].split('_')[0]
            output = open('../output/%s/%s_%s_output_%s.txt' % (output_dir, config['net_name'], tag, str(i_e + 1)), 'w')
            qid_uid_rel_score = {}
            qid_uid_score = {}
            genfun = generator.get_batch_generator()

            for input_data, y_true, curr_batch in genfun:
                y_pred = model.predict(input_data, batch_size=len(y_true))
                y_pred_reshape = np.reshape(y_pred, (len(y_pred),))
                # output the predict scores
                for (q, d, label), score in zip(curr_batch, y_pred_reshape):
                    output.write('%s\t%s\t%s\t%s\n' % (str(q), str(d), str(label), str(score)))

                    if q not in qid_uid_score:
                        qid_uid_score[q] = {}
                    qid_uid_score[q][d] = score

                    if q not in qid_uid_rel_score:
                        qid_uid_rel_score[q] = dict(label=list(), score=list())
                    qid_uid_rel_score[q]['label'].append(label)
                    qid_uid_rel_score[q]['score'].append(score)

            output.close()
            # calculate the metrices
            res = dict([[k, 0.] for k in eval_metrics.keys()])
            for k, eval_func in eval_metrics.items():
                for qid in qid_uid_rel_score:
                    res[k] += eval_func(y_true=qid_uid_rel_score[qid]['label'], y_pred=qid_uid_rel_score[qid]['score'])
                res[k] /= len(qid_uid_rel_score)

            if 'valid' not in tag:
                print '[%s]' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                print '[Eval] @ epoch: %d,' % (i_e + 1),
                print ', '.join(['%s: %.5f' % (k, res[k]) for k in res])
            else:
                # calculate the eval_loss
                all_pairs = generator.get_all_pairs()
                all_pairs_rel_score = {}
                for qid, dp_id, dn_id in all_pairs:
                    all_pairs_rel_score[(qid, dp_id, dn_id)] = {}
                    all_pairs_rel_score[(qid, dp_id, dn_id)]['score'] = [qid_uid_score[qid][dp_id],
                                                                         qid_uid_score[qid][dn_id]]
                    all_pairs_rel_score[(qid, dp_id, dn_id)]['rel'] = all_pairs[(qid, dp_id, dn_id)]

                eval_loss = cal_eval_loss(all_pairs_rel_score, tag, config['losses'])

                print '[%s]' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                print '[Eval] @ epoch: %d,' % (i_e + 1),
                print ', '.join(['%s: %.5f' % (k, eval_loss[k]) for k in eval_loss]),
                print ', '.join(['%s: %.5f' % (k, res[k]) for k in res])

        print ''


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='')
    parser.add_argument('--phase', default='train')
    parser.add_argument('--model_file', default='')
    args = parser.parse_args()

    if args.model_file == '':
        exit(0)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        KTF.set_session(sess)

        with open(args.model_file) as f:
            config = json.load(f)

        if args.phase == 'train':
            train(config)
        # elif args.phase == 'predict':
        #     predict(config)
        else:
            print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            print'Phase Error.'
            return
    print 'done'


if __name__ == '__main__':
    main(sys.argv)
