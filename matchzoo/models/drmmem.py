# -*- coding=utf-8 -*-
import random
import keras
import keras.backend as K
import time
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dense, Lambda, Activation, Permute, BatchNormalization, Dropout, Masking
from keras.layers import Reshape, Dot
from keras.activations import softmax, tanh
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
        doc = Input(name='doc', shape=(self.config['doc_maxlen'],))

        embedding = Embedding(self.config['vocab_size'],
                              self.config['embed_size'],
                              weights=[self.config['embed']],
                              trainable=self.embed_trainable)
        q_embed = embedding(query)
        d_embed = embedding(doc)
        cross = Dot(axes=[2, 2], normalize=True)([q_embed, d_embed])

        embedding_idf = Embedding(self.config['vocab_size'], 1,
                              weights=[self.config['idf_feat']],
                              trainable=self.embed_trainable)
        q_w_embed = embedding_idf(query)
        print q_w_embed.shape

        q_w = Dense(1, use_bias=False)(q_w_embed)
        print q_w.shape
        q_w = Reshape((self.config['query_maxlen'],))(q_w)
        print q_w.shape
        q_w = Activation('softmax')(q_w)
        print q_w.shape

        z = cross
        print z.shape
        z = Dropout(self.config['dropout_rate'])(z)
        for i in range(len(self.config['hidden_sizes_hist'])):
            z = Dense(self.config['hidden_sizes_hist'][i])(z)
            z = BatchNormalization(center=False, scale=False)(z)
            z = Activation('tanh')(z)

        print z.shape

        z = Reshape((self.config['query_maxlen'],))(z)

        out_ = Dot(axes= [1, 1])([z, q_w])

        model = Model(inputs=[query, doc], outputs=[out_])
        return model
