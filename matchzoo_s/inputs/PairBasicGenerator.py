class PairBasicGenerator(object):
    def __init__(self, config):
        self.__name = 'PairBasicGenerator'

        self.data_handler = open(config['data'])
        self.batch_size = config['batch_size']

    def __del__(self):
        self.data_handler.close()

    def get_batch(self):
        pass

    def get_batch_generator(self):
        pass