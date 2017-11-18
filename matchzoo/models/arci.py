# -*- coding=utf-8 -*-
import keras
import keras.backend as K
import time
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers import Reshape, Embedding,Merge, Dot
from keras.optimizers import Adam
from model import BasicModel


class ARCI(BasicModel):
    def __init__(self, config):
        super(ARCI, self).__init__(config)
        self.__name = 'ARCI'
        self.check_list = [ 'text1_maxlen', 'text2_maxlen',
                   'embed', 'embed_size', 'train_embed',  'vocab_size',
                   'kernel_size', 'kernel_count', 'dropout_rate',
                   'q_pool_size', 'd_pool_size']
        self.embed_trainable = config['train_embed']
        self.setup(config)
        if not self.check():
            raise TypeError('[ARCI] parameter check wrong')
        print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '[ARCI] init done'

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        self.set_default('kernel_count', 32)
        self.set_default('kernel_size', 3)
        self.set_default('q_pool_size', 2)
        self.set_default('d_pool_size', 2)
        self.set_default('dropout_rate', 0)
        self.config.update(config)

    def build(self):
        def mlp_work(input_dim):
            seq = Sequential()
            num_hidden_layers = len(self.config['hidden_sizes'])
            assert num_hidden_layers > 0
            if num_hidden_layers == 1:
                seq.add(Dense(self.config['hidden_sizes'][0], input_shape=(input_dim,)))
            else:
                seq.add(Dense(self.config['hidden_sizes'][0], activation='relu', input_shape=(input_dim,)))
                for i in range(num_hidden_layers - 2):
                    seq.add(Dense(self.config['hidden_sizes'][i+1], activation='relu'))
                seq.add(Dense(self.config['hidden_sizes'][-1]))
            return seq

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
        mlp = mlp_work(pool1_flat_drop.shape[1])
        out_ = mlp(pool1_flat_drop)
        # out_ = Dense(1)(pool1_flat_drop)


        #model = Model(inputs=[query, doc, dpool_index], outputs=out_)
        model = Model(inputs=[query, doc], outputs=out_)
        return model
