# -*- coding: utf8 -*-
import os
import sys
import time
import json
import argparse
import random
# random.seed(49999)
import numpy as np
# np.random.seed(49999)
import tensorflow as tf
# tf.set_random_seed(49999)
from collections import OrderedDict
from keras.models import Model
import keras.backend.tensorflow_backend as KTF

from utils import *
import inputs
import metrics
from losses import *


def load_model(config):
    global_conf = config["global"]
    model_type = global_conf['model_type']
    if model_type == 'JSON':
        mo = Model.from_config(config['model'])
    elif model_type == 'PY':
        model_config = config['model']
        model_config.update(config['inputs']['share'])
        sys.path.insert(0, model_config['model_path'])

        model = import_object(model_config['model_py'], model_config)
        mo = model.build()
    return mo


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
    if 'embed_path' in share_input_conf:
        embed_dict, vocab_size, embed_size, word_dict, idf_dict = read_embedding(share_input_conf['embed_path'])
        share_input_conf['word_dict'] = word_dict
        # share_input_conf['feat_size'] = vocab_size
        share_input_conf['vocab_size'] = vocab_size
        share_input_conf['embed_size'] = embed_size
        embed = np.float32(np.random.uniform(-4, 4, [vocab_size, embed_size]))
        embed_normalize = False
        if 'drmm' in config['model']['model_py'].lower():
            embed_normalize = True
        share_input_conf['embed'] = convert_embed_2_numpy('embed', embed_dict=embed_dict, embed=embed,
                                                          normalize=embed_normalize)

        idf = np.float32(np.random.uniform(3, 9, [vocab_size, 1]))
        share_input_conf['idf_feat'] = convert_embed_2_numpy('idf', embed_dict=idf_dict, embed=idf, normalize=False)

        print '[%s]' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '[Embedding] Embedding Load Done.'

    # if 'idf_feat' in share_input_conf:
    #     datapath = share_input_conf['idf_feat']
    #     idf_dict = read_idf(datapath, word_dict)
    #     idf = np.float32(np.random.uniform(1, 5, [vocab_size, 1]))
    #     config['inputs']['share']['idf_feat'] = convert_embed_2_numpy('idf', embed_dict=idf_dict, embed=idf, normalize=False)

    # if 'ngraph' in share_input_conf:
    #     datapath = share_input_conf['ngraph']
    #     ngraph, ngraph_size = read_ngraph(datapath)
    #     # new_ngraph = {}
    #     # new_ngraph[0] = []
    #     # for term in word_dict:
    #     #     new_ngraph[word_dict[term]] = []
    #     #     sharp_term = '#' + term + '#'
    #     #     for i in range(len(sharp_term)):
    #     #         for j in range(i+1, len(sharp_term)+1):
    #     #             part_term = sharp_term[i:j]
    #     #             if part_term in ngraph:
    #     #                 new_ngraph[word_dict[term]].append(ngraph[part_term])
    #     config['inputs']['share']['ngraph'] = ngraph # new_ngraph
    #     config['inputs']['share']['ngraph_size'] = ngraph_size


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
    print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '[Input] Process Input Tags. %s in TRAIN, %s in EVAL.' % (input_train_conf.keys(), input_eval_conf.keys())

    # initial data generator
    train_gen = OrderedDict()
    eval_gen = OrderedDict()

    for tag, conf in input_train_conf.items():
        generator = inputs.get(conf['input_type'])
        train_gen[tag] = generator( config = conf )

    for tag, conf in input_eval_conf.items():
        generator = inputs.get(conf['input_type'])
        eval_gen[tag] = generator( config = conf )

    ######### Load Model #########
    model = load_model(config)

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

    for i_e in range(global_conf['num_epochs']):
        # model.save_weights(weights_file)
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
            if config['net_name'].lower().startswith('duet_embed'):
                output_dir = '_'.join(config['net_name'].split('_')[:2])
            else:
                output_dir = config['net_name'].split('_')[0]
            output = open('../output/%s/%s_%s_output_%s.txt' % (output_dir, config['net_name'], tag, str(i_e+1)), 'w')
            qid_rel_uid = {}
            qid_uid_rel_score = {}
            genfun = generator.get_batch_generator()

            res = dict([[k,0.] for k in eval_metrics.keys()])
            # num_valid = 0

            for input_data, y_true, curr_batch in genfun:
                # curr_list = generator.get_currbatch()
                y_pred = model.predict(input_data, batch_size=len(y_true))
                y_pred = np.reshape(y_pred, (len(y_pred),))
                # output the predict scores
                # cnt = 0
                for (q, d, label), score in zip(curr_batch, y_pred):
                    output.write('%s\t%s\t%s\t%s\n' % (str(q), str(d), str(label), str(score)))
                    if q not in qid_rel_uid:
                        qid_rel_uid[q] = {}
                    if q not in qid_uid_rel_score:
                        qid_uid_rel_score[q] = dict(label=list(), score=list())
                    # for d in d_list:
                    if label not in qid_rel_uid[q]:
                        qid_rel_uid[q][label] = {}

                    qid_rel_uid[q][label][d] = score
                    qid_uid_rel_score[q]['label'].append(label)
                    qid_uid_rel_score[q]['score'].append(score)
            output.close()
            # generator.reset()
            # calculate the metrices
            for k, eval_func in eval_metrics.items():
                for qid in qid_uid_rel_score:
                    res[k] += eval_func(y_true=qid_uid_rel_score[qid]['label'], y_pred=qid_uid_rel_score[qid]['score'])
                res[k] /= len(qid_uid_rel_score)

            # calculate the eval_loss
            eval_loss = cal_eval_loss(qid_rel_uid, tag, input_eval_conf[tag], config['losses'])
            eval_res_list = eval_loss.items() + res.items()
            print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            print '[Eval] @ epoch: %d,' %(i_e+1), ', '.join(['%s: %.5f'%(k,v) for k, v in eval_res_list])+'\n'


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='')
    parser.add_argument('--phase', default='train') # , help='Phase: Can be train or predict, the default value is train.')
    parser.add_argument('--model_file') # , default='./models/matchzoo.model', help='Model_file: MatchZoo model file for the chosen model.')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    config = tf.ConfigProto()
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


if __name__ == '__main__':
    main(sys.argv)
