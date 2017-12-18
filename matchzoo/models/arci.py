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
        self.check_list = [ 'query_maxlen', 'doc_maxlen',
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

        self.config.update(config)

    def build(self):
        query = Input(name='query', shape=(self.config['query_maxlen'],))
        doc = Input(name='doc', shape=(self.config['doc_maxlen'],))
        # query_mask = Masking(mask_value=-1)(query)
        # doc_mask = Masking(mask_value=-1)(doc)

        embedding = Embedding(self.config['vocab_size'],
                              self.config['embed_size'],
                              weights=[self.config['embed']],
                              trainable = self.embed_trainable)

        q_embed = embedding(query)
        d_embed = embedding(doc)

        q_conv1 = Conv1D(self.config['kernel_count'],
                         self.config['kernel_size'],
                         activation='relu', padding='same') (q_embed)
        d_conv1 = Conv1D(self.config['kernel_count'],
                         self.config['kernel_size'],
                         activation='relu', padding='same') (d_embed)
        print q_conv1.shape
        print d_conv1.shape

        q_pool1 = MaxPooling1D(pool_size=self.config['q_pool_size']) (q_conv1)
        d_pool1 = MaxPooling1D(pool_size=self.config['d_pool_size']) (d_conv1)
        print q_pool1.shape
        print d_pool1.shape

        pool1 = Concatenate(axis=1) ([q_pool1, d_pool1])
        print pool1.shape

        pool1_flat = Flatten()(pool1)
        print pool1_flat.shape

        pool1_flat_drop = Dropout(rate=self.config['dropout_rate'])(pool1_flat)
        print pool1_flat_drop.shape

        hidden_res = pool1_flat_drop
        num_layers = len(self.config['hidden_sizes'])
        for i in range(num_layers):
            hidden_res = Dense(self.config['hidden_sizes'][i])(hidden_res)
            hidden_res = BatchNormalization(center=False, scale=False)(hidden_res)
            if i < num_layers - 1:
                hidden_res = Activation('relu')(hidden_res)
            else:
                hidden_res = Activation('tanh')(hidden_res)

        out_ = hidden_res

        model = Model(inputs=[query, doc], outputs=out_)
        return model
