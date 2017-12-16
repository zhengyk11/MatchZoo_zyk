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
        # hidden_layer = BatchNormalization(center=False, scale=False)(hidden_layer)
        # hidden_layer = Activation('tanh')(hidden_layer)
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
