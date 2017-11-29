# -*- coding=utf-8 -*-
import random
import keras
import keras.backend as K
import time
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dense, Activation, Merge, Lambda, Permute, BatchNormalization, Dropout
from keras.layers import Reshape, Dot
from keras.activations import softmax
from model import BasicModel

class DRMM(BasicModel):
    def __init__(self, config):
        super(DRMM, self).__init__(config)
        self._name = 'DRMM'
        self.check_list = [ 'text1_maxlen', 'hist_size',
                            'embed', 'embed_size', 'vocab_size',
                            'idf_feat', 'hidden_sizes_qw', 'hidden_sizes_hist', 'dropout_rate']
        self.setup(config)
        self.embed_trainable = config['train_embed']
        # self.initializer_fc = keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=11)
        # self.initializer_gate = keras.initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=11)
        if not self.check():
            raise TypeError('[DRMM] parameter check wrong')
        print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '[DRMM] init done'

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        # self.set_default('text1_maxlen', 5)
        # self.set_default('hist_size', 60)
        self.config.update(config)

    def build(self):
        # def tensor_product(x):
        #     a = x[0]
        #     b = x[1]
        #     y = K.batch_dot(a, b, axis=1)
        #     y = K.einsum('ijk, ikl->ijl', a, b)
        #     return y
        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        doc = Input(name='doc', shape=(self.config['text1_maxlen'], self.config['hist_size']))

        # embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable = self.embed_trainable)
        embedding = Embedding(self.config['vocab_size'], 1, weights=[self.config['idf_feat']], trainable=self.embed_trainable)
        q_embed = embedding(query)
        print q_embed.shape
        # q_w = Dense(1, kernel_initializer=self.initializer_gate, use_bias=False)(q_embed)
        # q_w = Dense(1, use_bias=False)(q_embed)

        q_w = q_embed
        for i in range(len(self.config['hidden_sizes_qw']) - 1):
            q_w = Dropout(self.config['dropout_rate'])(q_w)
            q_w = Dense(self.config['hidden_sizes_qw'][i], use_bias=False)(q_w)
            q_w = BatchNormalization()(q_w)
            q_w = Activation('relu')(q_w)
        q_w = Dropout(self.config['dropout_rate'])(q_w)
        q_w = Dense(self.config['hidden_sizes_qw'][-1], use_bias=False)(q_w)
        # q_w = BatchNormalization()(q_w)
        q_w = Activation('softmax')(q_w)
        # q_w = Lambda(lambda x: softmax(x, axis=1), output_shape=(self.config['text1_maxlen'], ))(q_w)
        print q_w.shape

        z = doc
        for i in range(len(self.config['hidden_sizes_hist'])-1):
            z  = Dropout(self.config['dropout_rate'])(z)
            z = Dense(self.config['hidden_sizes_hist'][i])(z) # kernel_initializer=self.initializer_fc
            z = BatchNormalization()(z)
            z = Activation('relu')(z)
        z = Dropout(self.config['dropout_rate'])(z)
        z = Dense(self.config['hidden_sizes_hist'][-1])(z) # kernel_initializer=self.initializer_fc
        z = BatchNormalization()(z)
        z = Activation('tanh')(z)
        print z.shape
        z = Permute((2, 1))(z)
        print z.shape
        z = Reshape((self.config['text1_maxlen'],))(z)
        print z.shape
        q_w = Reshape((self.config['text1_maxlen'],))(q_w)
        print q_w.shape

        out_ = Dot(axes= [1, 1])([z, q_w])
        print out_.shape

        model = Model(inputs=[query, doc], outputs=[out_])
        return model
