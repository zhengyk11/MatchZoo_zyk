# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
import six
import keras
from keras import backend as K
from keras.losses import *
from keras.layers import Lambda
from keras.utils.generic_utils import deserialize_keras_object
import tensorflow as tf

def cal_eval_loss(all_pairs_rel_score, tag, train_loss):
    res = {}
    if 'cross_entropy_loss' in train_loss:
        crossentropy_loss_list = dict(y_true=list(), y_pred=list())
        for qid, dp_id, dn_id in all_pairs_rel_score:

            dp_rel, dn_rel = all_pairs_rel_score[(qid, dp_id, dn_id)]['rel']
            dp_score, dn_score = all_pairs_rel_score[(qid, dp_id, dn_id)]['score']

            # dp_prob = np.exp(dp_rel)/np.sum(np.exp(dp_rel, dn_rel))
            dp_prob = dp_rel / (dp_rel + dn_rel)
            dn_prob = 1. - dp_prob

            # crossentropy_loss_list['y_true'].append([1., 0.])
            crossentropy_loss_list['y_true'].append([dp_prob, dn_prob])
            crossentropy_loss_list['y_pred'].append([dp_score, dn_score])

        exp_y_pred = np.exp(crossentropy_loss_list['y_pred'])
        sum_exp_y_pred = np.sum(exp_y_pred, axis=1)[:,None]
        softmax_y_pred = np.log(exp_y_pred / sum_exp_y_pred)
        sum_pred_true = np.sum(crossentropy_loss_list['y_true'] * softmax_y_pred, axis=1)
        crossentropy_loss = -1. * np.mean(sum_pred_true)
        res['%s_cross_entropy_loss' % tag] = crossentropy_loss

    if 'rank_hinge_loss' in train_loss:
        hinge_loss_list = []
        for qid, dp_id, dn_id in all_pairs_rel_score:
            dp_score, dn_score = all_pairs_rel_score[(qid, dp_id, dn_id)]['score']
            hinge_loss_list.append(max(0., 1. + dn_score - dp_score))
        res['%s_rank_hinge_loss' % tag] = np.mean(hinge_loss_list)

    return res

def cross_entropy_loss(target, output):
    target = tf.reshape(target, [-1,2])
    output = tf.reshape(output, [-1,2])
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=output))

def rank_hinge_loss(y_true, y_pred):
    y_pos = Lambda(lambda a: a[::2, :], output_shape= (1,))(y_pred)
    y_neg = Lambda(lambda a: a[1::2, :], output_shape= (1,))(y_pred)
    loss = K.maximum(0., 1. + y_neg - y_pos)
    return K.mean(loss)

def serialize(rank_loss):
    return rank_loss.__name__


def deserialize(name, custom_objects=None):
    return deserialize_keras_object(name,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='loss function')


def get(identifier):
    if identifier is None:
        return None
    if isinstance(identifier, six.string_types):
        identifier = str(identifier)
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret '
                         'loss function identifier:', identifier)
