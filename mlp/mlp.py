import numpy as np

class MLP(object):
    '''
    Multi-Layer Perceptron using minibatch variant of stochastic gradient descent
    '''
    def __init__(self, input_dim, hidden_dim, output_dim, batch_size=1000, eta=0.01, momentum=0.9, beta=1):
        '''
        TODO: Fix docs to naming conv.
        Args:
            inputs (int): number of input nodes
            hiddens (int): number of hidden layer nodes
            outputs (int): number of output nodes
            batch_size (int): size of batches for training
            eta (int): learning rate
            momentum (int): multiplicative dependency of previous gradient
            beta (int): constant B of sigmoid activation function 1 / (1 + exp(-Bx))
        '''
        #self.hidden_layer = Layer()
        #self.output_layer = Layer()
        self.input_dim = input_dim
        self.class_dim = output_dim
        self.l1 = np.random.uniform(low=-0.5, high=0.5, size=(input_dim + 1, hidden_dim))
        self.l2 = np.random.uniform(low=-0.5, high=0.5, size=(hidden_dim + 1, output_dim))
        self.batch_size = batch_size
        self.eta = eta
        self.momentum = momentum
        self.beta = beta or 1
        self.bias = 1
        pass
    
    def train(self, inputs, labels):
        '''
        TODO: Add documentation
        '''
        _l1_update = np.zeros(self.l1.shape)
        _l2_update = np.zeros(self.l2.shape)

        for _inputs, _targets in self.minibatch(inputs, labels):

            _inputs, _hidden_activations, _output_activations = self.run_forward(_inputs)

            #if t - o changes, so does the sign!
            _output_delta = _output_activations * (1 - _output_activations) * (_targets - _output_activations)
            #TODO: Verify first col of output_layer (i.e. [:, 1:]) should be trimmed
            _hidden_delta = _hidden_activations * (1 - _hidden_activations) * (np.dot(_output_delta, self.l2.T))

            print('hidden delta' , _hidden_delta)
            exit()
            _l1_update = (1 / self.batch_size) * self.eta*(np.dot(_inputs.T, _hidden_delta[:,1:])) + self.momentum*_l1_update
            _l2_update = (1 / self.batch_size) * self.eta*(np.dot(_hidden_activations.T, _output_delta)) + self.momentum*_l2_update
            self.l1 += _l1_update
            self.l2 += _l2_update

    def run_forward(self, inputs):
        '''
        Runs the input through the network returning each layers activations
        Args:
            inputs (np.array): of shape (N, input_dim)
        Returns:
            tuple: of ordered elements,
                (np.array): hidden activations of shape (N, self.hidden_dim) 
                (np.array): output activations of shape (N, self.output_dim)
        '''
        #print(self.l1, '\n', self.l2)
        _inputs = self._add_bias(inputs)
        #print('inputs with bais \n', _inputs)
        _hidden_activations = self._add_bias(self._activate(np.dot(_inputs, self.l1)))
        #print('hidden activations \n', _hidden_activations)
        _output_activations = self._activate(np.dot(_hidden_activations, self.l2))
        #print('output activations \n', _output_activations)
        return _inputs, _hidden_activations, _output_activations

    def _add_bias(self, inputs):
        '''
        Inserts self.bias column along inputs at x0
        Returns:
            np.array: Biased inputs, i.e. [[x1,x2,...,xn], ... ] => [[1,x1,x2,...,xn], ...]
        '''
        return np.concatenate((self.bias * np.ones((inputs.shape[0], 1)), inputs), axis=1)

    def _activate(self, layer):
        '''
        Returns:
            np.array: A sigmoid activation for each element in layer 
        '''
        return 1.0 / (1.0 + np.exp(-self.beta * layer))

    def minibatch(self, inputs, targets):
        start = 0
        while start < inputs.shape[0]:
            yield inputs[start:start + self.batch_size,:], targets[start:start+self.batch_size,:]
            start += self.batch_size

    def test(self, test_data, test_labels):
        '''
        Returns:
            (np.array): a confusion matrix for the test data of shape(class_dim, class_dim)
        '''
        #test_data = self._add_bias(test_data)
        confusion_matrix = np.zeros((self.class_dim, self.class_dim))

        for n in range(test_data.shape[0]):
            _input, _target = test_data[n].reshape(1, self.input_dim), test_labels[n].reshape(1, self.class_dim)
            _, _, _output = self.run_forward(_input)
            _prediction, _label = np.argmax(_output), np.argmax(_target)
            confusion_matrix[_prediction][_label] += 1
        return confusion_matrix


class Layer(object):

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
