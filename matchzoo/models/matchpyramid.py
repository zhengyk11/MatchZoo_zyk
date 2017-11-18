# -*- coding=utf-8 -*-
import keras
import keras.backend as K
import time
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers import Reshape, Embedding, Merge, Dot
from keras.optimizers import Adam
from model import BasicModel

import sys


sys.path.append('../matchzoo/layers/')
from DynamicMaxPooling import *


class MatchPyramid(BasicModel):
    def __init__(self, config):
        super(MatchPyramid, self).__init__(config)
        self.__name = 'MatchPyramid'
        self.check_list = [ 'text1_maxlen', 'text2_maxlen',
                   'embed', 'embed_size', 'vocab_size',
                   'kernel_size', 'kernel_count',
                   'dpool_size', 'dropout_rate', 'hidden_sizes']
        self.embed_trainable = config['train_embed']
        self.setup(config)
        if not self.check():
            raise TypeError('[MatchPyramid] parameter check wrong')
        print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '[MatchPyramid] init done'
        
    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)
            
        # self.set_default('kernel_count', 32)
        # self.set_default('kernel_size', [3, 3])
        # self.set_default('dpool_size', [3, 10])
        # self.set_default('dropout_rate', 0)
        self.config.update(config)

    def build(self):
        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        # dpool_index = Input(name='dpool_index', shape=[self.config['text1_maxlen'], self.config['text2_maxlen'], 3], dtype='int32')

        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable = self.embed_trainable)
        q_embed = embedding(query)
        d_embed = embedding(doc)

        cross = Dot(axes=[2, 2])([q_embed, d_embed])
        cross_reshape = Reshape((self.config['text1_maxlen'], self.config['text2_maxlen'], 1))(cross)

        conv2d = Conv2D(self.config['kernel_count'], self.config['kernel_size'], padding='same', activation='relu')
        maxpool = MaxPooling2D(pool_size=(self.config['dpool_size'][0], self.config['dpool_size'][1]))
        # dpool = DynamicMaxPooling(self.config['dpool_size'][0], self.config['dpool_size'][1])

        conv1 = conv2d(cross_reshape)
        # pool1 = dpool([conv1, dpool_index])
        pool1 = maxpool(conv1)
        pool1_flat = Flatten()(pool1)
        pool1_flat_drop = Dropout(rate=self.config['dropout_rate'])(pool1_flat)
        num_hidden_layers = len(self.config['hidden_sizes'])
        if num_hidden_layers == 1:
            out_ = Dense(self.config['hidden_sizes'][0], activation='tanh')(pool1_flat_drop)
        else:
            hidden_res = pool1_flat_drop
            for i in range(num_hidden_layers - 1):
                hidden_res = Activation('relu')(BatchNormalization()(Dense(self.config['hidden_sizes'][i])(hidden_res)))
            out_ = Dense(self.config['hidden_sizes'][-1], activation='tanh')(hidden_res)
        # out_ = Dense(1)(pool1_flat_drop)

        # model = Model(inputs=[query, doc, dpool_index], outputs=out_)
        model = Model(inputs=[query, doc], outputs=out_)
        return model
