

class Layer(object):
    '''
    Might be nice to use this in an mlp
    '''

    def __init__(self, inputs, outputs):
        self.weights = np.random.uniform(low=-0.5, high=0.5, size=(input_dim, output))
        self.weights_prev = np.zeros((inputs, outputs))
    def predict(self, inputs):
        inputs = self.add_bias(inputs)
        return self._activation(np.dot(inputs, self.weights))
    def _activation(self, inputs):
        pass
    def _add_bias(self, inputs):
        pass