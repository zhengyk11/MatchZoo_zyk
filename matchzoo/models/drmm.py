# -*- coding=utf-8 -*-
import random
import keras
import keras.backend as K
import time
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dense, Activation, Permute, BatchNormalization, Dropout, Masking
from keras.layers import Reshape, Dot
from keras.activations import softmax
from model import BasicModel

class DRMM(BasicModel):
    def __init__(self, config):
        super(DRMM, self).__init__(config)
        self._name = 'DRMM'
        self.check_list = [ 'query_maxlen', 'hist_size',
                            'embed', 'embed_size', 'vocab_size',
                            'idf_feat', 'hidden_sizes_qw',
                            'hidden_sizes_hist', 'dropout_rate']
        self.setup(config)
        self.embed_trainable = config['train_embed']
        # self.initializer_fc = keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=11)
        # self.initializer_gate = keras.initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=11)
        if not self.check():
            raise TypeError('[DRMM] parameter check wrong')
        print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        print '[DRMM] init done'

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        self.config.update(config)

    def build(self):
        query = Input(name='query', shape=(self.config['query_maxlen'],))
        doc = Input(name='doc', shape=(self.config['query_maxlen'], self.config['hist_size']))

        query_mask = Masking(mask_value=-1)(query)
        embedding = Embedding(self.config['vocab_size'], 1, weights=[self.config['idf_feat']], trainable=self.embed_trainable)
        q_embed = embedding(query_mask)
        print q_embed.shape


        # q_w = q_embed
        # q_w = Dropout(self.config['dropout_rate'])(q_w)
        # for i in range(len(self.config['hidden_sizes_qw']) - 1):
        #     q_w = Dense(self.config['hidden_sizes_qw'][i])(q_w)
        #     q_w = BatchNormalization()(q_w)
        #     q_w = Activation('relu')(q_w)
        # q_w = Dropout(self.config['dropout_rate'])(q_w)
        # q_w = Dense(self.config['hidden_sizes_qw'][-1])(q_w)
        # q_w = BatchNormalization()(q_w)
        q_w = Activation('tanh')(q_embed)
        # q_w = Lambda(lambda x: softmax(x, axis=1), output_shape=(self.config['query_maxlen'], ))(q_w)
        # print q_w.shape

        z = doc
        print z.shape
        for i in range(len(self.config['hidden_sizes_hist'])-1):
            z = Dense(self.config['hidden_sizes_hist'][i])(z) # kernel_initializer=self.initializer_fc
            z = BatchNormalization()(z)
            z = Activation('relu')(z)
        # z = Dropout(self.config['dropout_rate'])(z)
        z = Dense(self.config['hidden_sizes_hist'][-1])(z) # kernel_initializer=self.initializer_fc
        # z = BatchNormalization()(z)
        z = Activation('tanh')(z)
        print z.shape
        z = Permute((2, 1))(z)
        print z.shape
        z = Reshape((self.config['query_maxlen'],))(z)
        z = Dropout(self.config['dropout_rate'])(z)
        # print z.shape
        q_w = Reshape((self.config['query_maxlen'],))(q_w)
        q_w = Dropout(self.config['dropout_rate'])(q_w)
        print q_w.shape

        out_ = Dot(axes= [1, 1])([z, q_w])
        # print out_.shape

        model = Model(inputs=[query, doc], outputs=[out_])
        return model
