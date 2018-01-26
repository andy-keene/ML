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
        self.tolerance = 0.01

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
 
    def train(self, input_data, input_labels, epochs):
        '''
        Runs training regime according to the gradient descent variation of the perceptron learning 
        algorithm (PLA)

        Args:
            data (np.array): shape (N, input_dim)
            labels (np.array): shape (N, class_dim)
            epochs (int): number of times to iterate over input_data

        Returns:
            dict: dictionary of the accuracy for each epoch, i.e. { epoch_num: %accuracy, ... }
        '''
        input_data = self._add_bias(input_data)
        epoch_accuracy = dict()

        for epoch in range(epochs):
            confusion_matrix = np.zeros((self.class_dim, self.class_dim))
            for n in range(input_data.shape[0]):
                _input, _target = input_data[n].reshape(1, self.input_dim+1), input_labels[n].reshape(1, self.class_dim)
                _output = self.predict(_input)
                _prediction, _label = np.argmax(_output), np.argmax(_target)
                confusion_matrix[_prediction][_label] += 1

                #print('input:\n{}\noutput:\n{}\ntarget:\n{}'.format(_input, _output, _target))
                if _prediction != _label:
                    _activations = np.where(_output > 0, 1, 0)
                    self.weights -= self.eta*np.dot(_input.T, _activations - _target)
                #print('updated weights: {}\n\n\n'.format(self.weights))
            epoch_accuracy[epoch] = np.trace(confusion_matrix) / np.sum(confusion_matrix)
            print('finished epoch {} with %{}'.format(epoch, epoch_accuracy[epoch]))
        return epoch_accuracy

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

    def _train(self, input_data, input_labels, epochs):
        #this should be good now. just needs to be cleaned up
        #assumes exclusivity of class label (i.e. only [0,0,0,...,1,...])
        print('Weights size {} \n input_data size {}'.format(self.weights.shape, input_data.shape))
        print('Weights {}'.format(self.weights))
        input_data = self._add_bias(input_data)
        for n in range(input_data.shape[0]):

            x, t = input_data[n].reshape(1, self.input_dim+1), input_labels[n].reshape(1, self.class_dim)

            print('input ({}) {} \nlabel ({}) {}'.format(x.shape, x, t, t.shape))

            output = np.dot(x, self.weights)

            print('x . W ({}) {}'.format(output.shape, output))

            prediction, label = np.argmax(output), np.argmax(t)

            print('prediction: {}, label: {}'.format(prediction, label))
            
            #only if this was an incorrect prediction, update weights
            #if np.argmax(output) != np.argmax(t):
            activations = np.where(output > 0, 1, 0)
            print('activation ({}): {}, label ({}): {}'.format(activations.shape, activations, t.shape, t))
            print('x.T dot t - activations {}\n'.format(np.dot(x.T, t - activations)))

            # + x.T (dot) t - activations   || - x.T (dot) activations - t
            self.weights += self.eta*np.dot(x.T, t - activations)
            #print("x.T = {}, (t-output) = {}".format(x.shape, (t - output).shape))
            print('updated weights: {}\n\n\n'.format(self.weights))
        return