# -*- coding=utf-8 -*-
import keras
import keras.backend as K
import time
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers import Reshape, Embedding,Merge, Dot
from keras.optimizers import Adam
from model import BasicModel

import sys

sys.path.append('../matchzoo/layers/')
from DynamicMaxPooling import *
from Match import *

class ARCII(BasicModel):
    def __init__(self, config):
        super(ARCII, self).__init__(config)
        self.__name = 'ARCII'
        self.check_list = [ 'query_maxlen', 'doc_maxlen',
                            'embed', 'embed_size', 'vocab_size',
                            '1d_kernel_size', '1d_kernel_count',
                            'num_conv2d_layers','2d_kernel_sizes',
                            '2d_kernel_counts','2d_mpool_sizes',
                            'dropout_rate', 'hidden_sizes']
        self.embed_trainable = config['train_embed']
        self.setup(config)
        if not self.check():
            raise TypeError('[ARCII] parameter check wrong')
        print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        print '[ARCII] init done'

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        self.config.update(config)

    def build(self):

        query = Input(name='query', shape=(self.config['query_maxlen'],))
        doc = Input(name='doc', shape=(self.config['doc_maxlen'],))

        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']],
                              trainable = self.embed_trainable)
        q_embed = embedding(query)
        d_embed = embedding(doc)

        q_conv1 = Conv1D(self.config['1d_kernel_count'], self.config['1d_kernel_size'], padding='same') (q_embed)
        d_conv1 = Conv1D(self.config['1d_kernel_count'], self.config['1d_kernel_size'], padding='same') (d_embed)

        cross = Match(match_type='plus')([q_conv1, d_conv1])
        z = Reshape((self.config['query_maxlen'], self.config['doc_maxlen'], -1))(cross)

        for i in range(self.config['num_conv2d_layers']):
            z = Conv2D(self.config['2d_kernel_counts'][i], self.config['2d_kernel_sizes'][i], padding='same', activation='relu')(z)
            print z.shape
            z = MaxPooling2D(pool_size=(self.config['2d_mpool_sizes'][i][0], self.config['2d_mpool_sizes'][i][1]))(z)
            print z.shape

        pool1_flat = Flatten()(z)
        pool1_flat_drop = Dropout(rate=self.config['dropout_rate'])(pool1_flat)

        num_hidden_layers = len(self.config['hidden_sizes'])
        if num_hidden_layers == 1:
            out_ = Dense(self.config['hidden_sizes'][0], activation='tanh')(pool1_flat_drop)
        else:
            hidden_res = pool1_flat_drop
            for i in range(num_hidden_layers - 1):
                hidden_res = Dense(self.config['hidden_sizes'][i], activation='relu')(hidden_res)
            out_ = Dense(self.config['hidden_sizes'][-1], activation='tanh')(hidden_res)

        model = Model(inputs=[query, doc], outputs=out_)
        return model
