# -*- coding=utf-8 -*-
import keras
import keras.backend as K
import time
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers import Reshape, Embedding,Merge, Dot, BatchNormalization
from keras.optimizers import Adam
from model import BasicModel


class ARCI(BasicModel):
    def __init__(self, config):
        super(ARCI, self).__init__(config)
        self.__name = 'ARCI'
        self.check_list = [ 'text1_maxlen', 'text2_maxlen',
                   'embed', 'embed_size', 'train_embed',  'vocab_size',
                   'kernel_size', 'kernel_count', 'dropout_rate',
                   'q_pool_size', 'd_pool_size', 'hidden_sizes']
        self.embed_trainable = config['train_embed']
        self.setup(config)
        if not self.check():
            raise TypeError('[ARCI] parameter check wrong')
        print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '[ARCI] init done'

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        # self.set_default('kernel_count', 32)
        # self.set_default('kernel_size', 3)
        # self.set_default('q_pool_size', 2)
        # self.set_default('d_pool_size', 2)
        # self.set_default('dropout_rate', 0)
        # self.set_default('hidden_sizes', [300, 128])
        self.config.update(config)

    def build(self):
        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        #dpool_index = Input(name='dpool_index', shape=[self.config['text1_maxlen'], self.config['text2_maxlen'], 3], dtype='int32')

        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable = self.embed_trainable)
        q_embed = embedding(query)
        d_embed = embedding(doc)

        q_conv1 = Conv1D(self.config['kernel_count'], self.config['kernel_size'], padding='same') (q_embed)
        d_conv1 = Conv1D(self.config['kernel_count'], self.config['kernel_size'], padding='same') (d_embed)

        q_pool1 = MaxPooling1D(pool_size=self.config['q_pool_size']) (q_conv1)
        d_pool1 = MaxPooling1D(pool_size=self.config['d_pool_size']) (d_conv1)

        pool1 = Concatenate(axis=1) ([q_pool1, d_pool1])

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

        #model = Model(inputs=[query, doc, dpool_index], outputs=out_)
        model = Model(inputs=[query, doc], outputs=out_)
        return model
