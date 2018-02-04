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
        self.input_dim = input_dim
        self.class_dim = output_dim
        self.l1 = np.random.uniform(low=-0.5, high=0.5, size=(input_dim + 1, hidden_dim))
        self.l2 = np.random.uniform(low=-0.5, high=0.5, size=(hidden_dim + 1, output_dim))
        self.batch_size = batch_size
        self.eta = eta
        self.momentum = momentum
        self.beta = beta or 1
        self.bias = 1

    def run_training(self, dataset, epochs):
        '''
        Wrapper that runs the training regime specified by the project
        Args:
            dataset (dict): Contains the necessary training/test data and labels
                            i.e. {'train_data': (np.array), 'train_label': (np.array), ... }
            epochs (int): number of times to update the weight matrix according to sequential PLA
        Returns:
            dict: data describing the performance of the perceptron
        '''
        #for scoping issues w/ return :(
        train_confusion_matrix = None
        test_confusion_matrix = None
        epoch_accuracy = dict()

        #awkward for loop to save accuracy for pre-training
        for epoch in range(epochs):    
            train_confusion_matrix = self.test(dataset['train_data'], dataset['train_labels'])
            test_confusion_matrix = self.test(dataset['test_data'], dataset['test_labels'])
            epoch_accuracy[epoch] = {
                'train': np.trace(train_confusion_matrix) / np.sum(train_confusion_matrix),
                'test': np.trace(test_confusion_matrix) / np.sum(test_confusion_matrix)
            }
            #train network.
            self.train(dataset['train_data'], dataset['train_labels'])

        return {
            'accuracy' : epoch_accuracy,
            'confusion_matrix': test_confusion_matrix.tolist()
        }

    def train(self, inputs, labels):
        '''
        TODO: Add documentation
        '''
        inputs = self._add_bias(inputs)
        _l1_update = np.zeros(self.l1.shape)
        _l2_update = np.zeros(self.l2.shape)

        for _inputs, _targets in self.minibatch(inputs, labels):

            _hidden_activations, _output_activations = self.run_forward(_inputs)

            #if t - o changes, so does the sign!
            _output_delta = _output_activations * (1 - _output_activations) * (_targets - _output_activations)
            _hidden_delta = _hidden_activations * (1 - _hidden_activations) * (np.dot(_output_delta, self.l2.T))

            #note: must trim first col of _hidden_delta as it corresponds to hidden bias! Also will be 0 col. since (1-bias)*bias=0 for bias=1 
            _l1_update = (self.eta / self.batch_size) * np.dot(_inputs.T, _hidden_delta[:,1:]) + self.momentum*_l1_update
            _l2_update = (self.eta / self.batch_size) * np.dot(_hidden_activations.T, _output_delta) + self.momentum*_l2_update
            self.l1 += _l1_update
            self.l2 += _l2_update

    def test(self, inputs, labels):
        '''
        Tests the network
        Args:
            inputs (np.array): of shape (N, m)
            labels (np.array): of shape (N, self.class_dim)
        Returns:
            (np.array): a confusion matrix for the test data of shape(class_dim, class_dim)
        '''
        inputs = self._add_bias(inputs)
        _, _output = self.run_forward(inputs)
        return self._get_confusion_matrix(_output, labels)

    def run_forward(self, inputs):
        '''
        Runs the input through the network returning each layers activations
        Args:
            inputs (np.array): of shape (N, input_dim + 1), i.e. preprocessed with bias
        Returns:
            tuple: of ordered elements,
                (np.array): hidden activations of shape (N, self.hidden_dim + 1), i.e. with bias 
                (np.array): output activations of shape (N, self.output_dim)
        '''
        #print(self.l1, '\n', self.l2)
        #_inputs = self._add_bias(inputs)
        #print('inputs with bais \n', _inputs)
        _hidden_activations = self._add_bias(self._activate(np.dot(inputs, self.l1)))
        #print('hidden activations \n', _hidden_activations)
        _output_activations = self._activate(np.dot(_hidden_activations, self.l2))
        #print('output activations \n', _output_activations)
        return _hidden_activations, _output_activations

    def _add_bias(self, inputs):
        '''
        Inserts self.bias column along inputs at x0
        Returns:
            np.array: Biased inputs, i.e. [[x1,x2,...,xn], ... ] => [[1,x1,x2,...,xn], ...]
        '''
        return np.concatenate((self.bias * np.ones((inputs.shape[0], 1)), inputs), axis=1)

    def _activate(self, layer):
        '''
        Applies the sigmoid activation function to the given matrix
        Args:
            layer (np.array): output Matrix to be activated
        Returns:
            np.array: A sigmoid activation for each element in layer 
        '''
        return 1.0 / (1.0 + np.exp(-self.beta * layer))

    def minibatch(self, inputs, targets):
        '''
        Generates minibatch on the fly
        Args:
            inputs (np.array): 
            targets (np.array): 
        Returns:
            tuple: ordered elements upon __next__() call of
                np.array: next input batch of size (self.batch_size, inputs.shape[0])
                np.array: corresponding target batch of size (self.batch_size, targets.shape[0])                
        '''
        start = 0
        while start < inputs.shape[0]:
            yield inputs[start:start + self.batch_size,:], targets[start:start+self.batch_size,:]
            start += self.batch_size

    def _get_confusion_matrix(self, predictions, labels):
        '''
        Returns a confusion matrix for each prediction / label pair
        Args:
            predictions (np.array): of shape (N, x)
            labels (np.array): of shape (N, self.class_dim)
        Returns:
            np.array: confusion matrix of shape (self.class_dim, self.class_dim)
        '''
        _confusion_matrix = np.zeros((self.class_dim, self.class_dim))
        for n in range(predictions.shape[0]):
            _prediction, _label = np.argmax(predictions[n]), np.argmax(labels[n])
            _confusion_matrix[_prediction][_label] += 1
        return _confusion_matrix

