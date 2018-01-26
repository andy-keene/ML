import numpy as np
from perceptron import Perceptron
from dataset import Dataset

def main():
    
    mnist_data = Dataset.load_mnist()
    '''
    #perceptron = Perceptron(mnist_data["train_data"].shape[0], mnist_data["train_data"].shape[0], 0.001)
    perceptron = Perceptron(784, 10, 0.001)
    perceptron.train(mnist_data["train_data"], mnist_data["train_labels"], 0)
    '''

    for key in mnist_data:
        print('{} has shape {}'.format(key, mnist_data[key].shape))
    pass

    '''
    mnist_data = Dataset.load_mnist()['train_data'][0]
    print('input data shape {}'.format(mnist_data.shape))
    data = np.array([[0,0],[0,1],[1,0],[1,1]])
    labels = np.array([[0],[1],[1],[1]])
    perceptron = Perceptron(2, 1, 0.01)
    perceptron.train(data, labels, 1)
    '''

    #2 training samples, length 3
    input_data = np.random.randint(-5, 5, size=(2, 3))
    input_labels = np.array(
        [
            [0,1],
            [0,1],
            [1,0],
            [0,0],
            [0,0]
        ]
    )
    perceptron = Perceptron(3, 2, 0.01)
    perceptron.train(input_data, input_labels, 1)

if __name__ == '__main__':
    main()
