# -*- coding: utf8 -*-
import os
import sys
import time
import json
import argparse
import random
random.seed(49999)
import numpy
numpy.random.seed(49999)
import tensorflow
tensorflow.set_random_seed(49999)

from collections import OrderedDict

import keras
import keras.backend as K

from keras.models import Sequential, Model
import tensorflow as tf
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

    word_dict, vocab_size = read_word_dict_zyk(config['inputs']['share']['word_dict'])
    share_input_conf['feat_size'] = vocab_size
    share_input_conf['vocab_size'] = vocab_size
    share_input_conf['fill_word'] = 0

    # collect embedding
    if 'embed_path' in share_input_conf and 'embed_size' in share_input_conf:
        embed_dict = read_embedding(share_input_conf['embed_path'])
        embed = np.float32(np.random.uniform(-4, 4, [share_input_conf['vocab_size'], share_input_conf['embed_size']]))
        if 'drmm' in config['model']['model_py'].lower():
            share_input_conf['embed'] = convert_embed_2_numpy('embed', embed_dict=embed_dict, embed=embed, normalize=True)
        else:
            share_input_conf['embed'] = convert_embed_2_numpy('embed', embed_dict=embed_dict, embed=embed, normalize=False)

    elif 'embed_size' in share_input_conf:
        embed = np.float32(np.random.uniform(-4, 4, [share_input_conf['vocab_size'], share_input_conf['embed_size']]))
        share_input_conf['embed'] = embed

    print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '[Embedding] Embedding Load Done.'

    if 'idf_feat' in input_conf['share']:
        datapath = input_conf['share']['idf_feat']
        idf_dict = read_idf(datapath, word_dict)
        idf = np.float32(np.random.uniform(1, 5, [share_input_conf['vocab_size'], 1]))
        config['inputs']['share']['idf_feat'] = convert_embed_2_numpy('idf', embed_dict=idf_dict, embed=idf, normalize=False)


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
        # print conf
        # conf['data1'] = dataset['text1_corpus']
        # conf['data2'] = dataset['text2_corpus']
        if 'drmm' in config['model']['model_py'].lower():
            conf['hist_feats'] = cal_hist(conf)

        generator = inputs.get(conf['input_type'])
        train_gen[tag] = generator( config = conf )

    for tag, conf in input_eval_conf.items():
        # print conf
        # conf['data1'] = dataset['text1_corpus']
        # conf['data2'] = dataset['text2_corpus']
        if 'drmm' in config['model']['model_py'].lower():
            conf['hist_feats'] = cal_hist(conf)

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
        # print '[Train] @ %s epoch.' % i_e
        for tag, generator in train_gen.items():
            genfun = generator.get_batch_generator()
            # print '[Train] @ %s' % tag
            num_batch_cnt = 0
            for input_data, y_true in genfun:
                num_batch_cnt += 1
                info = model.fit(x=input_data, y=y_true, epochs=1, verbose=0)
                # y_pred = model.predict(x=input_data, batch_size=len(y_true))
                # print metrics.ndcg(10)(y_true, y_pred)
                print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '[Train] @ iter: %d,' % (i_e*num_batch+num_batch_cnt), 'train_%s: %.5f' %(config['losses'][0], info.history['loss'][0])
                if num_batch_cnt == num_batch:
                    break
            # model.fit_generator(
            #         genfun,
            #         steps_per_epoch = num_batch,
            #         epochs = 1,
            #         verbose = 2
            #     ) #callbacks=[eval_map])

        for tag, generator in eval_gen.items():
            if config['net_name'].lower().startswith('duet_embed'):
                output_dir = '_'.join(config['net_name'].split('_')[:2])
            else:
                output_dir = config['net_name'].split('_')[0]
            output = open('../output/%s/%s_%s_output_%s.txt' % (output_dir, config['net_name'], tag, str(i_e+1)), 'w')
            qid_rel_uid = {}
            qid_uid_rel_score = {}
            genfun = generator.get_batch_generator()
            # list_list = generator.get_list_list()
            # print '\n[Eval] @ %s ' % tag,
            res = dict([[k,0.] for k in eval_metrics.keys()])
            # num_valid = 0

            for input_data, y_true in genfun:
                curr_list = generator.get_currbatch()
                y_pred = model.predict(input_data, batch_size=len(y_true))
                # output the predict scores
                cnt = 0
                for q, d_list in curr_list:
                    if q not in qid_rel_uid:
                        qid_rel_uid[q] = {}
                    if q not in qid_uid_rel_score:
                        qid_uid_rel_score[q] = dict(rel=list(), score=list())
                    for d in d_list:
                        if d[0] not in qid_rel_uid[q]:
                            qid_rel_uid[q][d[0]] = {}
                        qid_rel_uid[q][d[0]][d[1]] = float(y_pred[cnt][0])
                        qid_uid_rel_score[q]['rel'].append(d[0])
                        qid_uid_rel_score[q]['score'].append(y_pred[cnt][0])
                        output.write('%s\t%s\t%s\t%s\n'%(str(q), str(d[1]), str(d[0]), str(y_pred[cnt][0])))
                        cnt += 1

                # calculate the metrices
                # if issubclass(type(generator), inputs.list_generator.ListBasicGenerator):
                #     list_counts = input_data['list_counts']
                #     for k, eval_func in eval_metrics.items():
                #         for lc_idx in range(len(list_counts)-1):
                #             pre = list_counts[lc_idx]
                #             suf = list_counts[lc_idx+1]
                #             res[k] += eval_func(y_true = y_true[pre:suf], y_pred = y_pred[pre:suf])
                #     num_valid += len(list_counts) - 1
                # else:
                #     for k, eval_func in eval_metrics.items():
                #         res[k] += eval_func(y_true = y_true, y_pred = y_pred)
                #     num_valid += 1
            generator.reset()
            output.close()
            # calculate the metrices
            for k, eval_func in eval_metrics.items():
                for qid in qid_uid_rel_score:
                    res[k] += eval_func(y_true=qid_uid_rel_score[qid]['rel'], y_pred=qid_uid_rel_score[qid]['score'])
                res[k] /= len(qid_uid_rel_score)


            # calculate the eval_loss
            eval_loss = cal_eval_loss(qid_rel_uid, tag, input_eval_conf[tag], config['losses'])
            eval_res_list = eval_loss.items() + res.items()
            print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '[Eval] @ epoch: %d,' %(i_e+1), ', '.join(['%s: %.5f'%(k,v) for k, v in eval_res_list])+'\n' # +',' ,', '.join(['%s: %.5f'%(k,v) for k, v in ]), '\n'
            sys.stdout.flush()


    model.save_weights(weights_file)

def cal_eval_loss(qid_rel_uid, tag, conf, train_loss):
    # print qid_rel_uid
    # exit(0)
    hinge_loss_list = []
    crossentropy_loss_list = dict(y_true=list(), y_pred=list())
    for q in qid_rel_uid:
        for hr in qid_rel_uid[q]:
            for lr in qid_rel_uid[q]:
                if hr <= lr:
                    continue
                if 'rel_gap' in conf and hr - lr <= conf['rel_gap']:
                    continue
                if 'high_label' in conf and hr < conf['high_label']:
                    continue
                for hu in qid_rel_uid[q][hr]:
                    for lu in qid_rel_uid[q][lr]:
                        crossentropy_loss_list['y_true'].append([1., 0.])
                        crossentropy_loss_list['y_pred'].append([qid_rel_uid[q][hr][hu], qid_rel_uid[q][lr][lu]])
                        hinge_loss_list.append(max(0., 1. + qid_rel_uid[q][lr][lu] - qid_rel_uid[q][hr][hu]))
    hinge_loss = sum(hinge_loss_list) / len(hinge_loss_list)
    crossentropy_loss = cross_entropy_loss(crossentropy_loss_list['y_true'], crossentropy_loss_list['y_pred']).eval()
    res = {}
    if 'cross_entropy_loss' in train_loss:
        res['%s_cross_entropy_loss'%tag]=crossentropy_loss
    if 'rank_hinge_loss' in train_loss:
        res['%s_rank_hinge_loss' % tag] = hinge_loss
    return res
    # return {'%s_hinge'%tag:hinge_loss, '%s_entropy'%tag:crossentropy_loss}

def predict(config):
    ######## Read input config ########
    word_dict, ngraphs = read_word_dict_zyk(config)
    print(json.dumps(config, indent=2))
    input_conf = config['inputs']
    share_input_conf = input_conf['share']

    # collect embedding 
    if 'embed_path' in share_input_conf:
        embed_dict = read_embedding(filename=share_input_conf['embed_path'], word_ids=None)
        # _PAD_ = share_input_conf['fill_word']
        # embed_dict[_PAD_] = np.zeros((share_input_conf['embed_size'], ), dtype=np.float32)
        embed = np.float32(np.random.uniform(-0.2, 0.2, [share_input_conf['vocab_size'], share_input_conf['embed_size']]))
        share_input_conf['embed'] = convert_embed_2_numpy(embed_dict, embed = embed)
    else:
        embed = np.float32(np.random.uniform(-0.2, 0.2, [share_input_conf['vocab_size'], share_input_conf['embed_size']]))
        share_input_conf['embed'] = embed
    print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '[Embedding] Embedding Load Done.'

    # list all input tags and construct tags config
    input_predict_conf = OrderedDict()
    for tag in input_conf.keys():
        if 'phase' not in input_conf[tag]:
            continue
        if input_conf[tag]['phase'] == 'PREDICT':
            input_predict_conf[tag] = {}
            input_predict_conf[tag].update(share_input_conf)
            input_predict_conf[tag].update(input_conf[tag])
    print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '[Input] Process Input Tags. %s in PREDICT.' % (input_predict_conf.keys())

    # collect dataset identification
    dataset = {}
    for tag in input_conf:
        if tag == 'share' or input_conf[tag]['phase'] == 'PREDICT':
            if 'text1_corpus' in input_conf[tag]:
                datapath = input_conf[tag]['text1_corpus']
                if datapath not in dataset:
                    dataset[datapath], _ = read_data(datapath, word_dict=word_dict)
            if 'text2_corpus' in input_conf[tag]:
                datapath = input_conf[tag]['text2_corpus']
                if datapath not in dataset:
                    dataset[datapath], _ = read_data(datapath, word_dict=word_dict)
    print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '[Dataset] %s Dataset Load Done.' % len(dataset)

    # initial data generator
    predict_gen = OrderedDict()

    for tag, conf in input_predict_conf.items():
        print conf
        conf['data1'] = dataset[conf['text1_corpus']]
        conf['data2'] = dataset[conf['text2_corpus']]
        generator = inputs.get(conf['input_type'])
        predict_gen[tag] = generator( 
                                    #data1 = dataset[conf['text1_corpus']],
                                    #data2 = dataset[conf['text2_corpus']],
                                     config = conf )  

    ######## Read output config ########
    output_conf = config['outputs']

    ######## Load Model ########
    global_conf = config["global"]
    weights_file = global_conf['weights_file']

    model = load_model(config)
    model.load_weights(weights_file)

    eval_metrics = OrderedDict()
    for mobj in config['metrics']:
        mobj = mobj.lower()
        if '@' in mobj:
            mt_key, mt_val = mobj.split('@', 1)
            eval_metrics[mobj] = metrics.get(mt_key)(int(mt_val))
        else:
            eval_metrics[mobj] = metrics.get(mobj)
    res = dict([[k,0.] for k in eval_metrics.keys()])

    for tag, generator in predict_gen.items():
        genfun = generator.get_batch_generator()
        print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '[Predict] @ %s ' % tag,
        num_valid = 0
        res_scores = {} 
        for input_data, y_true in genfun:
            list_counts = input_data['list_counts']
            y_pred = model.predict(input_data, batch_size=len(y_true) )

            for k, eval_func in eval_metrics.items():
                for lc_idx in range(len(list_counts)-1):
                    pre = list_counts[lc_idx]
                    suf = list_counts[lc_idx+1]
                    res[k] += eval_func(y_true = y_true[pre:suf], y_pred = y_pred[pre:suf])

            y_pred = np.squeeze(y_pred)
            for lc_idx in range(len(list_counts)-1):
                pre = list_counts[lc_idx]
                suf = list_counts[lc_idx+1]
                for p, y, t in zip(input_data['ID'][pre:suf], y_pred[pre:suf], y_true[pre:suf]):
                    if p[0] not in res_scores:
                        res_scores[p[0]] = {}
                    res_scores[p[0]][p[1]] = (y, t)

            num_valid += len(list_counts) - 1
        generator.reset()

        if tag in output_conf:
            if output_conf[tag]['save_format'] == 'TREC':
                with open(output_conf[tag]['save_path'], 'w') as f:
                    for qid, dinfo in res_scores.items():
                        dinfo = sorted(dinfo.items(), key=lambda d:d[1][0], reverse=True)
                        for inum,(did, (score, gt)) in enumerate(dinfo):
                            print >> f, '%s\tQ0\t%s\t%d\t%f\t%s'%(qid, did, inum, score, config['net_name'])
            elif output_conf[tag]['save_format'] == 'TEXTNET':
                with open(output_conf[tag]['save_path'], 'w') as f:
                    for qid, dinfo in res_scores.items():
                        dinfo = sorted(dinfo.items(), key=lambda d:d[1][0], reverse=True)
                        for inum,(did, (score, gt)) in enumerate(dinfo):
                            print >> f, '%s %s %s %s'%(gt, qid, did, score)

        print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '[Predict] results: ', '  '.join(['%s:%f'%(k,v/num_valid) for k, v in res.items()])
        sys.stdout.flush()


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='')
    parser.add_argument('--phase', default='train', help='Phase: Can be train or predict, the default value is train.')
    parser.add_argument('--model_file', default='./models/matchzoo.model', help='Model_file: MatchZoo model file for the chosen model.')
    args = parser.parse_args()
    model_file = args.model_file

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        KTF.set_session(sess)
        with open(model_file, 'r') as f:
            config = json.load(f)

        if args.phase == 'train':
            train(config)
        elif args.phase == 'predict':
            predict(config)
        else:
            print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'Phase Error.'
            return


if __name__ == '__main__':
    main(sys.argv)
