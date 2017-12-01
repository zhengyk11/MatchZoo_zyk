
# -*- coding=utf-8 -*-
import keras
import keras.backend as K
import time
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dense, Activation, Merge, Lambda, Permute
from keras.layers import Convolution1D, MaxPooling1D, Reshape, Dot
from keras.activations import softmax

from model import BasicModel

class CDSSM(BasicModel):
    def __init__(self, config):
        super(CDSSM, self).__init__(config)
        self.__name = 'CDSSM'
        self.check_list = [ 'query_maxlen', 'doc_maxlen',
                   'vocab_size', 'embed_size',
                   'kernel_count', 'kernel_size', 'hidden_sizes']
        self.embed_trainable = config['train_embed']
        self.setup(config)
        if not self.check():
            raise TypeError('[CDSSM] parameter check wrong')
        print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '[CDSSM] init done'

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        # self.set_default('hidden_sizes', [300, 128])
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
        query = Input(name='query', shape=(self.config['query_maxlen'],))
        doc = Input(name='doc', shape=(self.config['doc_maxlen'],))

        wordhashing = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable=self.embed_trainable)
        q_embed = wordhashing(query)
        d_embed = wordhashing(doc)
        conv1d = Convolution1D(self.config['kernel_count'], self.config['kernel_size'], padding='same', activation='relu')
        q_conv = conv1d(q_embed)
        d_conv = conv1d(d_embed)
        q_pool = MaxPooling1D(self.config['query_maxlen'])(q_conv)
        q_pool_re = Reshape((-1,))(q_pool)
        d_pool = MaxPooling1D(self.config['doc_maxlen'])(d_conv)
        d_pool_re = Reshape((-1,))(d_pool)


        # mlp = mlp_work(self.config['embed_size'])
        mlp = mlp_work(self.config['kernel_count'])

        rq = mlp(q_pool_re)
        rd = mlp(d_pool_re)
        #out_ = Merge([rq, rd], mode='cos', dot_axis=1)
        out_ = Dot( axes= [1, 1], normalize=True)([rq, rd])

        model = Model(inputs=[query, doc], outputs=[out_])
        return model
