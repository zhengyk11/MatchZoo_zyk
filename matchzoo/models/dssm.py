# -*- coding=utf-8 -*-
import time
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Flatten
from keras.layers import Reshape, Dot

from model import BasicModel


class DSSM(BasicModel):
    def __init__(self, config):
        super(DSSM, self).__init__(config)
        self.__name = 'DSSM'
        self.check_list = ['ngraph_size', 'hidden_sizes']
        self.setup(config)
        if not self.check():
            raise TypeError('[DSSM] parameter check wrong')
        print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        print '[DSSM] init done'

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        self.config.update(config)

    def build(self):
        query = Input(name='query', shape=(self.config['ngraph_size'],))
        doc = Input(name='doc', shape=(self.config['ngraph_size'],))

        def mlp_work(input_dim):
            seq = Sequential()
            num_hidden_layers = len(self.config['hidden_sizes'])
            assert num_hidden_layers > 0
            seq.add(Dense(self.config['hidden_sizes'][0], activation='relu', input_shape=(input_dim,)))
            for i in range(1, num_hidden_layers):
                seq.add(Dense(self.config['hidden_sizes'][i]))
                if i < num_hidden_layers - 1:
                    seq.add(Activation(activation='relu'))
            return seq

        # query_flat = Flatten()(query)
        # doc_flat = Flatten()(doc)

        mlp = mlp_work(self.config['ngraph_size'])

        rq = mlp(query)
        rd = mlp(doc)
        out_ = Dot(axes=[1, 1], normalize=True)([rq, rd])

        model = Model(inputs=[query, doc], outputs=[out_])
        return model
