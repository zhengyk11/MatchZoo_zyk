# -*- coding: utf8 -*-
import os
import sys
import json
import argparse

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

word_dict = {}
word_embed_list = []
inverse_id_dict = {}

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

    # collect dataset identification
    global word_embed_list
    dataset = {}
    invalid_idf = 0.
    for tag in input_conf['share']:
        # if tag != 'share' and input_conf[tag]['phase'] == 'PREDICT':
        #     continue
        if 'text1_corpus' in tag:
            datapath = input_conf['share'][tag]
            # if datapath not in dataset:
            data, data_word = read_data(datapath, word_dict=word_dict)
            word_embed_list += data_word
            if 'text1_corpus' not in dataset:
                dataset['text1_corpus'] = data
            else:
                dataset['text1_corpus'].update(data)
        if 'text2_corpus' in tag:
            datapath = input_conf['share'][tag]
            # if datapath not in dataset:
            data, data_word = read_data(datapath, word_dict=word_dict)
            word_embed_list += data_word
            if 'text2_corpus' not in dataset:
                dataset['text2_corpus'] = data
            else:
                dataset['text2_corpus'].update(data)
        if 'idf_feat' in tag:
            datapath = input_conf['share'][tag]
            data = read_idf(datapath, word_dict=word_dict)
            invalid_idf = data[-1]
            if 'idf_feat' not in dataset:
                dataset['idf_feat'] = data

    word_embed_list = sorted(list(set(word_embed_list)))
    len_word_embed_list = len(word_embed_list)
    global inverse_id_dict
    for i, j in enumerate(word_embed_list):
        inverse_id_dict[j] = i
    inverse_id_dict[-1] = len_word_embed_list

    new_idf_dict = {}
    #
    for d in dataset:
        if 'text' in d:
            for tid in dataset[d]:
                for i, j in enumerate(dataset[d][tid]):
                    dataset[d][tid][i] = inverse_id_dict[j]
        elif 'idf' in d and 'drmm' in config['model']['model_py'].split('.')[0].lower():
            for i, j in enumerate(word_embed_list):
                if j in dataset[d]:
                    new_idf_dict[i] = [dataset[d][j]]
                else:
                    new_idf_dict[i] = [dataset[d][-1]]

            print 'idf feat size: %s' % len(new_idf_dict)
            dataset['idf_feat'] = convert_embed_2_numpy(new_idf_dict, max_size=len_word_embed_list+1)
            dataset['idf_feat'][-1] = np.array([invalid_idf], dtype=np.float32)# np.float32(np.random.uniform(-0.2, 0.2, [1]))
            config['inputs']['share']['idf_feat'] = dataset['idf_feat']
    inverse_id_dict.pop(-1)
    print '[Dataset] %s Dataset Load Done.' % len(dataset)
    ##

    # collect embedding

    share_input_conf['fill_word'] = len_word_embed_list
    share_input_conf['vocab_size'] = len_word_embed_list + 1
    share_input_conf['feat_size'] = len_word_embed_list + 1
    config['inputs']['share']['feat_size'] = len_word_embed_list + 1 # can delete, the same effect as last code
    if 'embed_path' in share_input_conf:
        embed_dict = read_embedding(filename=share_input_conf['embed_path'], word_ids=inverse_id_dict)
        embed_dict[share_input_conf['fill_word']] = np.zeros((share_input_conf['embed_size'], ), dtype=np.float32)
        embed = np.float32(np.random.uniform(-0.2, 0.2, [share_input_conf['vocab_size'], share_input_conf['embed_size']]))
        share_input_conf['embed'] = convert_embed_2_numpy(embed_dict, embed = embed)
    else:
        embed = np.float32(np.random.uniform(-0.2, 0.2, [share_input_conf['vocab_size'], share_input_conf['embed_size']]))
        share_input_conf['embed'] = embed
    print '[Embedding] Embedding Load Done.'

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
    print '[Input] Process Input Tags. %s in TRAIN, %s in EVAL.' % (input_train_conf.keys(), input_eval_conf.keys())

    # collect dataset identification
    # dataset = {}
    # for tag in input_conf:
    #     if tag != 'share' and input_conf[tag]['phase'] == 'PREDICT':
    #         continue
    #     if 'text1_corpus' in input_conf[tag]:
    #         datapath = input_conf[tag]['text1_corpus']
    #         if datapath not in dataset:
    #             dataset[datapath], _ = read_data(datapath)
    #     if 'text2_corpus' in input_conf[tag]:
    #         datapath = input_conf[tag]['text2_corpus']
    #         if datapath not in dataset:
    #             dataset[datapath], _ = read_data(datapath)
    # print '[Dataset] %s Dataset Load Done.' % len(dataset)

    # initial data generator
    train_gen = OrderedDict()
    eval_gen = OrderedDict()


    for tag, conf in input_train_conf.items():
        # print conf
        conf['data1'] = dataset['text1_corpus']
        conf['data2'] = dataset['text2_corpus']
        # if 'idf_feat' in dataset:
        #     config['idf_feat'] = dataset['idf_feat']
        generator = inputs.get(conf['input_type'])
        train_gen[tag] = generator( config = conf )

    for tag, conf in input_eval_conf.items():
        # print conf
        conf['data1'] = dataset['text1_corpus']
        conf['data2'] = dataset['text2_corpus']
        # if 'idf_feat' in dataset:
        #     config['idf_feat'] = dataset['idf_feat']
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
    print '[Model] Model Compile Done.\n'

    for i_e in range(global_conf['num_epochs']):
        # print '[Train] @ %s epoch.' % i_e
        for tag, generator in train_gen.items():
            genfun = generator.get_batch_generator()
            # print '[Train] @ %s' % tag
            num_batch_cnt = 0
            for input_data, y_true in genfun:
                num_batch_cnt += 1
                if num_batch_cnt > num_batch:
                    break
                info = model.fit(x=input_data, y=y_true, epochs=1, verbose=0)
                # y_pred = model.predict(x=input_data, batch_size=len(y_true))
                # print metrics.ndcg(10)(y_true, y_pred)
                print '[Train] @ iter: %d,' % (i_e*num_batch+num_batch_cnt-1), 'loss: %.4f' %info.history['loss'][0]
            # model.fit_generator(
            #         genfun,
            #         steps_per_epoch = num_batch,
            #         epochs = 1,
            #         verbose = 2
            #     ) #callbacks=[eval_map])

        for tag, generator in eval_gen.items():
            output = open('../output/%s/%s_%s_output_%s.txt' % (config['net_name'].split('_')[0], config['net_name'], tag, str(i_e)), 'w')
            qid_rel_uid = {}
            genfun = generator.get_batch_generator()
            list_list = generator.get_list_list()
            # print '\n[Eval] @ %s ' % tag,
            res = dict([[k,0.] for k in eval_metrics.keys()])
            num_valid = 0

            for input_data, y_true in genfun:
                y_pred = model.predict(input_data, batch_size=len(y_true))
                # output the predict scores
                cnt = 0
                for q, d_list in list_list:
                    if q not in qid_rel_uid:
                        qid_rel_uid[q] = {}
                    for d in d_list:
                        if d[0] not in qid_rel_uid[q]:
                            qid_rel_uid[q][d[0]] = {}
                        qid_rel_uid[q][d[0]][d[1]] = float(y_pred[cnt][0])
                        output.write('%s %s %s %s\n'%(str(q), str(d[1]), str(int(d[0])), str(y_pred[cnt][0])))
                        cnt += 1
                # calculate the metrices
                if issubclass(type(generator), inputs.list_generator.ListBasicGenerator):
                    list_counts = input_data['list_counts']
                    for k, eval_func in eval_metrics.items():
                        for lc_idx in range(len(list_counts)-1):
                            pre = list_counts[lc_idx]
                            suf = list_counts[lc_idx+1]
                            res[k] += eval_func(y_true = y_true[pre:suf], y_pred = y_pred[pre:suf])
                    num_valid += len(list_counts) - 1
                else:
                    for k, eval_func in eval_metrics.items():
                        res[k] += eval_func(y_true = y_true, y_pred = y_pred)
                    num_valid += 1
            generator.reset()
            # calculate the eval_loss
            eval_loss = cal_eval_loss(qid_rel_uid, input_eval_conf[tag])
            print '[Eval] @ epoch: %d,' %( i_e ), ', '.join(['%s: %.4f'%(k,v) for k, v in eval_loss.items()]), ',' ,', '.join(['%s: %.4f'%(k,v/num_valid) for k, v in res.items()]), '\n'
            sys.stdout.flush()
            output.close()

    # model.save_weights(weights_file)

def cal_eval_loss(qid_rel_uid, conf):
    hinge_loss_list = []
    crossentropy_loss_list = dict(y_true=list(), y_pred=list())
    for q in qid_rel_uid:
        for hr in qid_rel_uid[q]:
            for lr in qid_rel_uid[q]:
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
    with tf.Session() as sess:
        crossentropy_loss = my_categorical_crossentropy(crossentropy_loss_list['y_true'], crossentropy_loss_list['y_pred']).eval()
    return dict(RankHingeLoss=hinge_loss, CrossEntropyLoss=crossentropy_loss)

def predict(config):
    ######## Read input config ########

    print(json.dumps(config, indent=2))
    input_conf = config['inputs']
    share_input_conf = input_conf['share']

    # collect embedding 
    if 'embed_path' in share_input_conf:
        embed_dict = read_embedding(filename=share_input_conf['embed_path'], word_ids=inverse_id_dict)
        _PAD_ = share_input_conf['fill_word']
        embed_dict[_PAD_] = np.zeros((share_input_conf['embed_size'], ), dtype=np.float32)
        embed = np.float32(np.random.uniform(-0.02, 0.02, [share_input_conf['vocab_size'], share_input_conf['embed_size']]))
        share_input_conf['embed'] = convert_embed_2_numpy(embed_dict, embed = embed)
    else:
        embed = np.float32(np.random.uniform(-0.2, 0.2, [share_input_conf['vocab_size'], share_input_conf['embed_size']]))
        share_input_conf['embed'] = embed
    print '[Embedding] Embedding Load Done.'

    # list all input tags and construct tags config
    input_predict_conf = OrderedDict()
    for tag in input_conf.keys():
        if 'phase' not in input_conf[tag]:
            continue
        if input_conf[tag]['phase'] == 'PREDICT':
            input_predict_conf[tag] = {}
            input_predict_conf[tag].update(share_input_conf)
            input_predict_conf[tag].update(input_conf[tag])
    print '[Input] Process Input Tags. %s in PREDICT.' % (input_predict_conf.keys())

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
    print '[Dataset] %s Dataset Load Done.' % len(dataset)

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
        print '[Predict] @ %s ' % tag,
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

        print '[Predict] results: ', '  '.join(['%s:%f'%(k,v/num_valid) for k, v in res.items()])
        sys.stdout.flush()


def read_word_dict_zyk(config):
    global word_dict
    word_dict_filepath = config['inputs']['share']['word_dict']
    with open(word_dict_filepath) as f:
        for line in f:
            w, id = line[:-1].split('\t')
            word_dict[w] = int(id)


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
    sess = tf.Session(config=config)
    KTF.set_session(sess)
    with open(model_file, 'r') as f:
        config = json.load(f)
    phase = args.phase

    read_word_dict_zyk(config)

    if args.phase == 'train':
        train(config)
    elif args.phase == 'predict':
        predict(config)
    else:
        print 'Phase Error.'
    return


if __name__ == '__main__':
    main(sys.argv)
