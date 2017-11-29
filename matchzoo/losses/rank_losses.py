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

def cal_eval_loss(qid_rel_uid, tag, conf, train_loss):
    res = {}
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
                        if 'cross_entropy_loss' in train_loss:
                            crossentropy_loss_list['y_true'].append([1., 0.])
                            crossentropy_loss_list['y_pred'].append([qid_rel_uid[q][hr][hu], qid_rel_uid[q][lr][lu]])
                        if 'rank_hinge_loss' in train_loss:
                            hinge_loss_list.append(max(0., 1. + qid_rel_uid[q][lr][lu] - qid_rel_uid[q][hr][hu]))
    if 'rank_hinge_loss' in train_loss:
        hinge_loss = sum(hinge_loss_list) / len(hinge_loss_list)
        res['%s_rank_hinge_loss' % tag] = hinge_loss
    if 'cross_entropy_loss' in train_loss:
        crossentropy_loss = cross_entropy_loss(crossentropy_loss_list['y_true'], crossentropy_loss_list['y_pred']).eval()
        res['%s_cross_entropy_loss' % tag] = crossentropy_loss
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
