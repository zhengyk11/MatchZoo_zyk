# -*- coding=utf-8 -*-
import keras
import keras.backend as K
import time
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers import Reshape, Embedding,Merge, Dot, BatchNormalization, Permute
from keras.optimizers import Adam
from model import BasicModel


class DUETEM(BasicModel):
    def __init__(self, config):
        super(DUETEM, self).__init__(config)
        self.__name = 'DUETEM'
        self.check_list = [ 'query_maxlen', 'doc_maxlen',
                            'kernel_count', 'local_kernel_size', 'local_mpool_size',
                            'local_hidden_sizes', 'dist_query_kernel_sizes',
                            'dist_doc_kernel_sizes', 'dist_query_mpool_sizes','dist_doc_mpool_sizes',
                            'dist_hidden_sizes', 'dropout_rate', 'hidden_sizes']
        self.embed_trainable = config['train_embed']
        self.setup(config)
        # if not self.check():
        #     raise TypeError('[DUETEM] parameter check wrong')
        print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '[DUETEM] init done'

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        self.config.update(config)

    def build(self):

        # def mlp_work(input_dim):
        #     seq = Sequential()
        #     num_hidden_layers = len(self.config['hidden_sizes'])
        #     assert num_hidden_layers > 0
        #     if num_hidden_layers == 1:
        #         seq.add(Dense(self.config['hidden_sizes'][0], input_shape=(input_dim,)))
        #     else:
        #         seq.add(Dense(self.config['hidden_sizes'][0], activation='relu', input_shape=(input_dim,)))
        #         for i in range(num_hidden_layers - 2):
        #             seq.add(Dense(self.config['hidden_sizes'][i+1], activation='relu'))
        #         seq.add(Dense(self.config['hidden_sizes'][-1]))
        #     return seq
            # seq = Sequential()
            # num_hidden_layers = len(self.config['hidden_sizes'])
            # for i in range(num_hidden_layers):
            #     if i == 0:
            #         seq.add(Dropout(self.config['dropout_rate'], input_shape=(input_dim,)))
            #     else:
            #         seq.add(Dropout(self.config['dropout_rate']))
            #     seq.add(Dense(self.config['hidden_sizes'][i]))
            #     seq.add(BatchNormalization())
            #     seq.add(Activation(activation='relu'))
            #     # if i < num_hidden_layers - 1:
            #     #     seq.add(Activation(activation='relu'))
            #     # else:
            #     #     seq.add(Activation(activation='tanh'))
            # return seq

        local_feat = Input(name='local_feat', shape=(self.config['query_maxlen'], self.config['doc_maxlen'],))
        query = Input(name='query', shape=(self.config['query_maxlen'],))
        doc = Input(name='doc', shape=(self.config['doc_maxlen'],))
        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']],
                              trainable=self.embed_trainable)
        q_embed = embedding(query)
        print 1, q_embed.shape
        d_embed = embedding(doc)
        print 2, d_embed.shape

        local_feat_reshape = Reshape((self.config['query_maxlen'], self.config['doc_maxlen'], 1))(local_feat)
        print 3, local_feat_reshape.shape
        local_feat_conv = Conv2D(self.config['kernel_count'], [1, self.config['doc_maxlen']],
                       activation='tanh')(local_feat_reshape)
        print 4, local_feat_conv.shape
        local_feat_conv_reshape = Reshape([self.config['query_maxlen'], -1])(local_feat_conv)
        print 5, local_feat_conv_reshape.shape
        local_feat_conv_reshape = Permute((2, 1))(local_feat_conv_reshape)
        print 6, local_feat_conv_reshape.shape
        hidden_layer = local_feat_conv_reshape
        hidden_layer = Dense(1, activation='tanh')(hidden_layer)
        print 7, hidden_layer.shape
        hidden_layer = Reshape([self.config['kernel_count'],])(hidden_layer)
        print 8, hidden_layer.shape
        hidden_layer = Dense(self.config['kernel_count'], activation='tanh')(hidden_layer)
        print 9, hidden_layer.shape
        hidden_layer = Dropout(self.config['dropout_rate'])(hidden_layer)
        print 10, hidden_layer.shape
        hidden_layer = Dense(1, activation='tanh')(hidden_layer)
        print 11, hidden_layer.shape
        local_out_ = hidden_layer

        q_embed = Reshape((self.config['query_maxlen'], self.config['embed_size'], 1))(q_embed)
        print 12, q_embed.shape
        q_conv = Convolution2D(self.config['kernel_count'],
                               [self.config['dist_kernel_size'], self.config['embed_size']],
                               activation='tanh')(q_embed)
        print 13, q_conv.shape
        q_conv = Reshape([-1, self.config['kernel_count'], 1])(q_conv)
        print 14, q_conv.shape
        q_pool = MaxPooling2D([self.config['query_maxlen'] - self.config['dist_kernel_size']+1, 1], strides=(1,1))(q_conv)
        print 15, q_pool.shape
        q_pool = Reshape([-1,])(q_pool)
        print 16, q_pool.shape
        rq = Dense(self.config['kernel_count'], activation='tanh')(q_pool)
        print 17, rq.shape

        d_embed = Reshape((self.config['doc_maxlen'], self.config['embed_size'], 1))(d_embed)
        print 18, d_embed.shape
        d_conv = Convolution2D(self.config['kernel_count'],
                               [self.config['dist_kernel_size'], self.config['embed_size']],
                               activation='tanh')(d_embed)
        print 19, d_conv.shape
        d_conv = Reshape([-1, self.config['kernel_count'], 1])(d_conv)
        print 20, d_conv.shape
        d_pool = MaxPooling2D([self.config['pooling_kernel_width_doc'], 1], strides=(1,1))(d_conv)
        print 21, d_pool.shape
        d_conv = Convolution2D(self.config['kernel_count'],
                               [1, self.config['kernel_count']],
                               activation='tanh')(d_pool)
        print 22, d_conv.shape
        rd = Reshape([-1, self.config['kernel_count']])(d_conv)
        print 23, rd.shape

        # q_pool_re = Reshape((-1,))(q_pool)
        # d_pool = MaxPooling1D(self.config['doc_maxlen'])(d_conv)
        # d_pool_re = Reshape((-1,))(d_pool)

        # rd = mlp(d_pool_re)

        dist_out_ = Multiply()([rq, rd])
        print 24, dist_out_.shape
        dist_out_ = Permute((2, 1))(dist_out_)
        print 25, dist_out_.shape
        dist_out_ = Dense(1, activation='tanh')(dist_out_)
        print 26, dist_out_.shape

        dist_out_ = Reshape([-1,])(dist_out_)
        print 27, dist_out_.shape
        dist_out_ = Dense(self.config['kernel_count'], activation='tanh')(dist_out_)
        print 28, dist_out_.shape
        dist_out_ = Dropout(self.config['dropout_rate'])(dist_out_)
        print 29, dist_out_.shape
        dist_out_ = Dense(1, activation='tanh')(dist_out_)
        print 30, dist_out_.shape


        # dist_out_ = Dot(axes= [1, 1], normalize=True)([rq, rd])
        # merge local and dist scores
        # local_dist_out = Concatenate(axis=1)([local_out_, dist_out_])
        # print local_dist_out.shape
        out_ = Add()([local_out_, dist_out_])
        print 31, out_.shape
        # out_ = Dense(1, activation='tanh')(local_dist_out)
        # print out_.shape

        model = Model(inputs=[local_feat, query, doc], outputs=out_)
        return model

















        # print embed_q.shape
        # print embed_d.shape
        #
        # # local mdel
        # local_feat = Dot(axes=[2, 2], normalize=True)([embed_q, embed_d])
        # local_out_reshape = Reshape([self.config['query_maxlen'], self.config['doc_maxlen'], 1])(local_feat)
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
        # # q = Reshape([self.config['query_maxlen'], self.config['embed_size'], 1])(embed_q)
        # # for i in range(len(self.config['dist_query_kernel_sizes'])):
        # #     q = Conv2D(self.config['kernel_count'], self.config['dist_query_kernel_sizes'][i], padding='same', activation='relu')(q)
        # #     q = MaxPooling2D(pool_size=self.config['dist_query_mpool_sizes'][i])(q)
        # # embed_q_flatten = Flatten()(q)
        # # print embed_q_flatten.shape
        # # embed_q_out = Dense(self.config['kernel_count'], activation='relu')(embed_q_flatten)
        # # print embed_q_out.shape
        # #
        # # # dist_doc
        # # d = Reshape([self.config['doc_maxlen'], self.config['embed_size'], 1])(embed_d)
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
