import numpy as np

class Perceptron(object):
    '''
    Implements a simple Perceptron network using sequential training
    '''

    def __init__(self, input_size, class_size, eta):
        '''
        Args:
            eta (int): learning rate
            input_size (int): length of input, assumes shape = (input_size, )
            class_size (int): number of classes
        '''
        self.input_dim = input_size
        self.class_dim = class_size
        self.weights = np.random.uniform(low=-0.5, high=0.5, size=(input_size + 1, class_size))
        self.confusion_matrix = np.zeros((class_size, class_size))
        self.accuracy = 0
        self.eta = eta
        self.bias = 1
        #tolerance of 0.001 %
        self.tolerance = 0.00001

    def _add_bias(self, input_data):
        '''
        Inserts self.bias column along inputs at x0
        Returns:
            np.array: Biased input_data, i.e. [[x1,x2,...,xn], ... ] => [[1,x1,x2,...,xn], ...]
        '''
        return np.concatenate((self.bias * np.ones((input_data.shape[0], 1)), input_data), axis=1)

    def predict(self, data):
        '''
        Args:
            data (np.array): shape(N, input_dim)
        Returns:
            np.array: perceptron output vector for data, shape(1, class_dim)
        '''
        return np.dot(data, self.weights)

    def run_training(self, dataset, epochs):
        '''TODO: set up algorithm for data collection
        Wrapper that runs the training regime specified by the project

        Args:
            dataset (dict): Contains the necessary training/test data and labels
                            i.e. {'train_data': (np.array), 'train_label': (np.array), ... }
            epochs (int): number of times to update the weight matrix according to sequential PLA
        Returns:
            dict: data describing the performance of the perceptron
        '''
        #for scoping issues w/ return :(
        train_confusion_matrix, test_confusion_matrix = None, None
        epoch_accuracy = dict()
        #initial accuracy check with anthony?
        for epoch in range(epochs):
            train_confusion_matrix = self.train(dataset['train_data'], dataset['train_labels'])
            test_confusion_matrix = self.test(dataset['test_data'], dataset['test_labels'])
            epoch_accuracy[epoch] = {
                'train': np.trace(train_confusion_matrix) / np.sum(train_confusion_matrix),
                'test': np.trace(test_confusion_matrix) / np.sum(test_confusion_matrix)
            }

            if epoch > 1 and epoch_accuracy[epoch]['test'] < epoch_accuracy[epoch - 1]['test'] + self.tolerance:
                break
        return {
            'accuracy' : epoch_accuracy,
            'confusion_matrix': test_confusion_matrix.tolist()
        }

    def train(self, input_data, input_labels):
        '''
        Runs training regime according to sequential update PLA for one epoch

        Args:
            data (np.array): shape (N, input_dim)
            labels (np.array): shape (N, class_dim)

        Returns:
            dict: dictionary of the accuracy for each epoch, i.e. { epoch_num: %accuracy, ... }
            np.array: confusion matrix for the training set of shape (class_dim, class_dim)
        '''
        input_data = self._add_bias(input_data)
        confusion_matrix = np.zeros((self.class_dim, self.class_dim))

        for n in range(input_data.shape[0]):
            _input, _target = input_data[n].reshape(1, self.input_dim+1), input_labels[n].reshape(1, self.class_dim)
            _output = self.predict(_input)
            _prediction, _label = np.argmax(_output), np.argmax(_target)
            confusion_matrix[_prediction][_label] += 1
            if _prediction != _label:
                _activations = np.where(_output > 0, 1, 0)
                self.weights -= self.eta*np.dot(_input.T, _activations - _target)
        return confusion_matrix

    def test(self, test_data, test_labels):
        '''
        Returns:
            (np.array): a confusion matrix for the test data of shape(class_dim, class_dim)
        '''
        test_data = self._add_bias(test_data)
        confusion_matrix = np.zeros((self.class_dim, self.class_dim))

        for n in range(test_data.shape[0]):
            _input, _target = test_data[n].reshape(1, self.input_dim+1), test_labels[n].reshape(1, self.class_dim)
            _output = self.predict(_input)
            _prediction, _label = np.argmax(_output), np.argmax(_target)
            confusion_matrix[_prediction][_label] += 1
        return confusion_matrix
