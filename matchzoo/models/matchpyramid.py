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
        self.check_list = [ 'query_maxlen', 'doc_maxlen',
                            'embed', 'embed_size', 'vocab_size',
                            'kernel_size', 'kernel_count',
                            'dpool_size', 'dropout_rate', 'hidden_sizes']
        self.embed_trainable = config['train_embed']
        self.setup(config)
        if not self.check():
            raise TypeError('[MatchPyramid] parameter check wrong')
        print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        print '[MatchPyramid] init done'

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        self.config.update(config)

    def build(self):
        query = Input(name='query', shape=(self.config['query_maxlen'],))
        doc = Input(name='doc', shape=(self.config['doc_maxlen'],))

        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']],
                              trainable=self.embed_trainable)
        q_embed = embedding(query)
        d_embed = embedding(doc)

        cross = Dot(axes=[2, 2], normalize=True)([q_embed, d_embed])
        cross_reshape = Reshape((self.config['query_maxlen'], self.config['doc_maxlen'], 1))(cross)

        conv1 = Conv2D(self.config['kernel_count'], self.config['kernel_size'], padding='same', activation='relu')(cross_reshape)
        pool1 = MaxPooling2D(pool_size=(self.config['dpool_size'][0], self.config['dpool_size'][1]))(conv1)
        pool1_flat = Flatten()(pool1)

        pool1_flat_dropout = Dropout(self.config['dropout_rate'])(pool1_flat)

        num_hidden_layers = len(self.config['hidden_sizes'])
        hidden_layer = pool1_flat_dropout
        for i in range(num_hidden_layers):
            # hidden_layer =
            hidden_layer = Dense(self.config['hidden_sizes'][i])(hidden_layer)
            hidden_layer = BatchNormalization()(hidden_layer)
            if i < num_hidden_layers - 1:
                hidden_layer = Activation('relu')(hidden_layer)
            else:
                out_ = Activation('tanh')(hidden_layer)

        # out_ = Dense(self.config['hidden_sizes'][-1], activation='tanh')(hidden_res)
        # out_ = Dense(1)(pool1_flat_drop)

        # model = Model(inputs=[query, doc, dpool_index], outputs=out_)
        model = Model(inputs=[query, doc], outputs=out_)
        return model
