# -*- coding=utf-8 -*-
import random
import keras
import keras.backend as K
import time
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dense, Activation, Merge, Lambda, Permute
from keras.layers import Reshape, Dot
from keras.activations import softmax
from model import BasicModel

class DRMM(BasicModel):
    def __init__(self, config):
        super(DRMM, self).__init__(config)
        self._name = 'DRMM'
        self.check_list = [ 'text1_maxlen', 'hist_size',
                'embed', 'embed_size', 'vocab_size',
                'idf_feat', 'hidden_sizes']
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

        self.set_default('text1_maxlen', 5)
        self.set_default('hist_size', 60)
        self.config.update(config)

    def build(self):
        def tensor_product(x):
            a = x[0]
            b = x[1]
            y = K.batch_dot(a, b, axis=1)
            y = K.einsum('ijk, ikl->ijl', a, b)
            return y
        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        doc = Input(name='doc', shape=(self.config['text1_maxlen'], self.config['hist_size']))

        # embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable = self.embed_trainable)
        embedding = Embedding(self.config['vocab_size'], 1, weights=[self.config['idf_feat']], trainable=self.embed_trainable)
        q_embed = embedding(query)
        # q_w = Dense(1, kernel_initializer=self.initializer_gate, use_bias=False)(q_embed)
        q_w = Dense(1, use_bias=False)(q_embed)
        q_w = Lambda(lambda x: softmax(x, axis=1), output_shape=(self.config['text1_maxlen'], ))(q_w)
        z = doc
        for i in range(len(self.config['hidden_sizes'])):
            # z = Dense(self.config['hidden_sizes'][i], kernel_initializer=self.initializer_fc)(z)
            z = Dense(self.config['hidden_sizes'][i])(z)
            z = Activation('tanh')(z)
        z = Permute((2, 1))(z)
        z = Reshape((self.config['text1_maxlen'],))(z)
        q_w = Reshape((self.config['text1_maxlen'],))(q_w)

        out_ = Dot( axes= [1, 1])([z, q_w])

        model = Model(inputs=[query, doc], outputs=[out_])
        return model
