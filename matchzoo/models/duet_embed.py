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
                            'kernel_count', 'local_kernel_size', 'local_mpool_size',
                            'local_hidden_sizes', 'dist_query_kernel_sizes',
                            'dist_doc_kernel_sizes', 'dist_query_mpool_sizes','dist_doc_mpool_sizes',
                            'dist_hidden_sizes', 'dropout_rate', 'hidden_sizes']
        self.embed_trainable = config['train_embed']
        self.setup(config)
        if not self.check():
            raise TypeError('[DUET_EMBED] parameter check wrong')
        print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '[DUET_EMBED] init done'

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        self.config.update(config)

    def build(self):

        def mlp_work(input_dim):
            seq = Sequential()
            num_hidden_layers = len(self.config['hidden_sizes'])
            for i in range(num_hidden_layers):
                if i == 0:
                    seq.add(Dropout(self.config['dropout_rate'], input_shape=(input_dim,)))
                else:
                    seq.add(Dropout(self.config['dropout_rate']))
                seq.add(Dense(self.config['hidden_sizes'][i]))
                seq.add(BatchNormalization())
                seq.add(Activation(activation='relu'))
                # if i < num_hidden_layers - 1:
                #     seq.add(Activation(activation='relu'))
                # else:
                #     seq.add(Activation(activation='tanh'))
            return seq

        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']],
                              trainable=self.embed_trainable)
        q_embed = embedding(query)
        d_embed = embedding(doc)

        cross = Dot(axes=[2, 2], normalize=True)([q_embed, d_embed])
        cross_reshape = Reshape((self.config['text1_maxlen'], self.config['text2_maxlen'], 1))(cross)

        conv1 = Conv2D(self.config['kernel_count'], self.config['local_kernel_size'], padding='same',
                       activation='relu')(cross_reshape)
        pool1 = MaxPooling2D(pool_size=self.config['local_mpool_size'])(conv1)
        pool1_flat = Dense(self.config['kernel_count'])(Flatten()(pool1))

        local_mlp = mlp_work(self.config['kernel_count'])
        local_out_ = local_mlp(pool1_flat)

        conv1d = Convolution1D(self.config['kernel_count'], self.config['kernel_size'], padding='same', activation='relu')
        q_conv = conv1d(q_embed)
        d_conv = conv1d(d_embed)
        q_pool = MaxPooling1D(self.config['text1_maxlen'])(q_conv)
        q_pool_re = Reshape((-1,))(q_pool)
        d_pool = MaxPooling1D(self.config['text2_maxlen'])(d_conv)
        d_pool_re = Reshape((-1,))(d_pool)

        mlp = mlp_work(self.config['kernel_count'])

        rq = mlp(q_pool_re)
        rd = mlp(d_pool_re)

        dist_out_ = Dot(axes= [1, 1], normalize=True)([rq, rd])
        # merge local and dist scores
        local_dist_out = Concatenate(axis=1)([local_out_, dist_out_])
        print local_dist_out.shape
        # out_ = Add()([local_out, dist_out])
        out_ = Dense(1, activation='tanh')(local_dist_out)
        print out_.shape

        model = Model(inputs=[query, doc], outputs=dist_out_)
        return model

















        # print embed_q.shape
        # print embed_d.shape
        #
        # # local mdel
        # local_feat = Dot(axes=[2, 2], normalize=True)([embed_q, embed_d])
        # local_out_reshape = Reshape([self.config['text1_maxlen'], self.config['text2_maxlen'], 1])(local_feat)
        # local_conv2d = Conv2D(self.config['kernel_count'], self.config['local_kernel_size'], padding='same', activation='relu')(local_out_reshape)
        # local_conv2d_flatten = Flatten()(local_conv2d)
        #
        # num_hidden_layers = len(self.config['local_hidden_sizes'])
        # hidden_res = local_conv2d_flatten
        # for i in range(num_hidden_layers):
        #     hidden_res = Dropout(self.config['dropout_rate'])(hidden_res)
        #     hidden_res = Dense(self.config['local_hidden_sizes'][i])(hidden_res)
        #     hidden_res = BatchNormalization()(hidden_res)
        #     hidden_res = Activation('relu')(hidden_res)
        # local_out = hidden_res
        #
        # # # dist_query
        # # q = Reshape([self.config['text1_maxlen'], self.config['embed_size'], 1])(embed_q)
        # # for i in range(len(self.config['dist_query_kernel_sizes'])):
        # #     q = Conv2D(self.config['kernel_count'], self.config['dist_query_kernel_sizes'][i], padding='same', activation='relu')(q)
        # #     q = MaxPooling2D(pool_size=self.config['dist_query_mpool_sizes'][i])(q)
        # # embed_q_flatten = Flatten()(q)
        # # print embed_q_flatten.shape
        # # embed_q_out = Dense(self.config['kernel_count'], activation='relu')(embed_q_flatten)
        # # print embed_q_out.shape
        # #
        # # # dist_doc
        # # d = Reshape([self.config['text2_maxlen'], self.config['embed_size'], 1])(embed_d)
        # # for i in range(len(self.config['dist_doc_kernel_sizes'])):
        # #     d = Conv2D(self.config['kernel_count'], self.config['dist_doc_kernel_sizes'][i], padding='same', activation='relu')(d)
        # #     d = MaxPooling2D(pool_size=self.config['dist_doc_mpool_sizes'][i])(d)
        # # embed_d_reshape = Reshape([-1, self.config['kernel_count']])(d)
        # # # embed_d_flatten = Flatten()(d)
        # # embed_d_out = Dense(self.config['kernel_count'], activation='relu')(embed_d_reshape)
        # # print embed_d_out.shape
        # #
        # # # hardamard product
        # # dist_input = Multiply()([embed_q_out, embed_d_out])
        # # print dist_input.shape
        # #
        # # # dist model
        # # dist_input_flatten = Flatten()(dist_input)
        # # print dist_input_flatten.shape
        # #
        # # num_hidden_layers = len(self.config['dist_hidden_sizes'])
        # # hidden_res = dist_input_flatten
        # # for i in range(num_hidden_layers):
        # #     hidden_res = Dropout(self.config['dropout_rate'])(hidden_res)
        # #     hidden_res = Dense(self.config['dist_hidden_sizes'][i])(hidden_res)
        # #     hidden_res = BatchNormalization()(hidden_res)
        # #     hidden_res = Activation('relu')(hidden_res)
        # # dist_out = hidden_res
        # # print "dist_out", dist_out.shape
        # #
        # # # merge local and dist scores
        # # local_dist_out = Concatenate(axis=1)([local_out, dist_out])
        # # print local_dist_out.shape
        # # # out_ = Add()([local_out, dist_out])
        # # out_ = Dense(1, activation='tanh')(local_dist_out)
        # # print out_.shape
        #
        # model = Model(inputs=[query, doc], outputs=local_out)
        # return model
        #
