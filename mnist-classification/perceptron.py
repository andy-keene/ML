
class Perceptron(object):
    import numpy as np
    '''
        Implements a simple Perceptron network using sequential training

    '''

    def __init__(self, input_size, class_size, eta):
        '''
            eta (int): learning rate
            input_size (int): length of input, assumes shape = (input_size, )
            class_size (int): number of classes
        '''
        self.weights = np.random.uniform(low = -0.5, high = 0.5, size=(input_size + 1, class_size))
        self.confusion_matrix = np.zeros((class_size, class_size))
        self.accuracy = 0
        self.eta = eta
        self.bias = 1

    def train(self, input_data, input_labels, epochs):
        pass

    def test(self, test_data, test_labels):
        pass