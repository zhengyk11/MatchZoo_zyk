# -*- coding=utf-8 -*-
import keras
import keras.backend as K
import time
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers import Reshape, Embedding,Merge, Dot, BatchNormalization
from keras.optimizers import Adam
from model import BasicModel
# import sys
# sys.path.append('../matchzoo/layers/')
# from Match import *

class DUET_EMBED(BasicModel):
    def __init__(self, config):
        super(DUET_EMBED, self).__init__(config)
        self.__name = 'DUET_EMBED'
        self.check_list = [ 'text1_maxlen', 'text2_maxlen',
                            'kernel_count', 'local_kernel_size', 'dropout_rate',
                            'dist_kernel_size', 'pooling_kernel_width_doc',
                            'dist_doc_kernel_size']
        self.embed_trainable = config['train_embed']
        self.setup(config)
        if not self.check():
            raise TypeError('[DUET_EMBED] parameter check wrong')
        print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '[DUET_EMBED] init done'

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
        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']],
                              trainable=self.embed_trainable)
        embed_q = embedding(query)
        embed_d = embedding(doc)

        local_feat = Dot(axes=[2, 2], normalize=True)([embed_q, embed_d])
        # local_feat = Input(name='local_feats', shape=(self.config['text1_maxlen'],self.config['text2_maxlen'],))
        # query = Input(name='query', shape=(self.config['num_ngraphs'], self.config['text1_maxlen'],))
        # doc = Input(name='doc', shape=(self.config['num_ngraphs'], self.config['text2_maxlen'],))

        print local_feat.shape
        print embed_q.shape
        print embed_d.shape
        # num_hidden_nodes = self.config['kernel_count']

        word_window_size = self.config['dist_kernel_size'][0] #3
        pooling_kernel_width_query = self.config['text1_maxlen'] - word_window_size + 1  # = 8
        pooling_kernel_width_doc = self.config['pooling_kernel_width_doc'] #100
        # num_pooling_windows_doc = ((self.config['text2_maxlen'] - word_window_size + 1) - pooling_kernel_width_doc) + 1  # = 899
        print word_window_size, pooling_kernel_width_query, pooling_kernel_width_doc

        # duet_local = Sequential()
        # duet_local.add(Conv2D(self.config['kernel_count'], self.config['local_kernel_size'], activation='tanh',
        #                       input_shape=(self.config['text1_maxlen'], self.config['text2_maxlen'])))
        # duet_local.add(Dense(self.config['kernel_count'], activation='tanh'))
        # duet_local.add(Dense(self.config['kernel_count'], activation='tanh'))
        # duet_local.add(Dropout(self.config['dropout_rate']))
        # duet_local.add(Dense(1, activation='tanh'))
        # local_out = duet_local(local_feat)
        local_out = Reshape([self.config['text1_maxlen'], self.config['text2_maxlen'], 1])(local_feat)
        local_out = Conv2D(self.config['kernel_count'], self.config['local_kernel_size'], activation='tanh')(local_out)
        # , input_shape=(self.config['text1_maxlen'], self.config['text2_maxlen']))(local_feat)
        print local_out.shape
        local_out = Flatten()(local_out)
        print local_out.shape
        local_out = Dense(self.config['kernel_count'], activation='tanh')(local_out)
        print local_out.shape
        local_out = Dense(self.config['kernel_count'], activation='tanh')(local_out)
        print local_out.shape
        # local_out = Flatten()(local_out)
        # print local_out.shape
        local_out = Dropout(self.config['dropout_rate'])(local_out)
        print local_out.shape
        local_out = Dense(1, activation='tanh')(local_out)
        print local_out.shape

        # duet_embed_q = Sequential()
        # duet_embed_q.add(Conv2D(self.config['kernel_count'], self.config['dist_kernel_size'], activation='tanh')
                               #, input_shape=(self.config['num_ngraphs'], self.config['text1_maxlen'])))
        # duet_embed_q.add(MaxPool2D((1, pooling_kernel_width_query)))
        # duet_embed_q.add(Dense(self.config['kernel_count'], activation='tanh'))

        # embed_q = duet_embed_q(query)
        embed_q = Reshape([self.config['text1_maxlen'], self.config['embed_size'], 1])(embed_q)
        embed_q = Conv2D(self.config['kernel_count'], self.config['dist_kernel_size'], activation='tanh')(embed_q)
        print embed_q.shape
        embed_q = MaxPool2D((pooling_kernel_width_query, 1), strides=(1,1))(embed_q)
        print embed_q.shape
        embed_q = Flatten()(embed_q)
        print embed_q.shape
        embed_q = Dense(self.config['kernel_count'], activation='tanh')(embed_q)
        print embed_q.shape

        # duet_embed_d = Sequential()
        # duet_embed_d.add(Conv2D(self.config['kernel_count'], self.config['dist_kernel_size'], activation='tanh')
        #                        #, input_shape=(self.config['num_ngraphs'], self.config['text2_maxlen'])))
        # duet_embed_d.add(MaxPool2D((1, pooling_kernel_width_doc)))
        # duet_embed_d.add(Dense(self.config['kernel_count'], activation='tanh'))
        #
        # embed_d = duet_embed_d(doc)
        embed_d = Reshape([self.config['text2_maxlen'], self.config['embed_size'], 1])(embed_d)
        embed_d = Conv2D(self.config['kernel_count'], self.config['dist_kernel_size'], activation='tanh')(embed_d)
        print embed_d.shape
        embed_d = MaxPool2D((pooling_kernel_width_doc, 1), strides=(1,1))(embed_d)
        print embed_d.shape
        embed_d = Reshape([-1, self.config['kernel_count'], 1])(embed_d)
        print 'reshape', embed_d.shape
        embed_d = Conv2D(self.config['kernel_count'], self.config['dist_doc_kernel_size'], activation='tanh')(embed_d)
        print embed_d.shape
        embed_d = Reshape([-1, self.config['kernel_count']])(embed_d)
        print embed_d.shape

        dist_input = Multiply()([embed_q, embed_d])
        # dist_input = [embed_q*x for x in ]
        # dist_input = Match(match_type='mul')([embed_q, K.m K.transpose(embed_d)])
        print dist_input.shape
        dist_input = Flatten()(dist_input)
        print dist_input.shape
        # duet_distrib = Sequential()
        # duet_distrib.add(Dense(self.config['kernel_count'], activation='tanh'))
        # duet_distrib.add(Dense(self.config['kernel_count'], activation='tanh'))
        # duet_distrib.add(Dropout(self.config['dropout_rate']))
        # duet_distrib.add(Dense(1, activation='tanh'))
        # dist_out = duet_distrib(dist_input)

        dist_out = Dense(self.config['kernel_count'], activation='tanh')(dist_input)
        print dist_out.shape
        dist_out = Dense(self.config['kernel_count'], activation='tanh')(dist_out)
        print dist_out.shape
        dist_out = Dropout(self.config['dropout_rate'])(dist_out)
        print dist_out.shape
        dist_out = Dense(1, activation='tanh')(dist_out)
        print dist_out.shape

        local_dist_out = Concatenate(axis=1)([local_out, dist_out])
        # print local_dist_out.shape
        # out_ = Add()([local_out, dist_out])
        out_ = Dense(1, activation='tanh')(local_dist_out)
        print out_.shape
        model = Model(inputs=[query, doc], outputs=out_)
        return model
        # net_local = [C.slice(features_local, 0, idx, idx + 1) for idx in range(0, num_docs)]
        # net_local = [C.reshape(d, (1, words_per_query, words_per_doc)) for d in net_local]
        # net_local = [duet_local(d) for d in net_local]
        # net_local = [C.reshape(d, (1, 1)) for d in net_local]
        # net_local = C.splice(net_local)
        #
        # net_distrib_q = C.reshape(features_distrib_query, (num_ngraphs, words_per_query, 1))
        # net_distrib_q = duet_embed_q(net_distrib_q)
        # net_distrib_q = [net_distrib_q for idx in range(0, num_pooling_windows_doc)]
        # net_distrib_q = C.splice(net_distrib_q)
        # net_distrib_q = C.reshape(net_distrib_q, (num_hidden_nodes * num_pooling_windows_doc, 1))
        #
        # net_distrib_d = [C.slice(features_distrib_docs, 0, idx, idx + 1) for idx in range(0, num_docs)]
        # net_distrib_d = [C.reshape(d, (num_ngraphs, words_per_doc, 1)) for d in net_distrib_d]
        # net_distrib_d = [duet_embed_d(d) for d in net_distrib_d]
        # net_distrib_d = [C.reshape(d, (num_hidden_nodes * num_pooling_windows_doc, 1)) for d in net_distrib_d]
        #
        # net_distrib = [C.element_times(net_distrib_q, d) for d in net_distrib_d]
        # net_distrib = [duet_distrib(d) for d in net_distrib]
        # net_distrib = [C.reshape(d, (1, 1)) for d in net_distrib]
        # net_distrib = C.splice(net_distrib)

        # net = C.plus(net_local, net_distrib)

        #model = Model(inputs=[query, doc, dpool_index], outputs=out_)

