from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

class Dataset(object):    
    def load_mnist():
        '''
        Saves:
            train_data (np.array):  of shape (60000, 784)
            train_labels (np.array): of shape (60000, 10)
            test_data (np.array): of shape (10000, 784)
            test_labels (np.array): of shape (1000, 10)
        '''
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        train_data = np.vstack((mnist.train._images, mnist.validation._images))
        train_labels = np.vstack((mnist.train._labels, mnist.validation._labels))
        test_data = mnist.test._images
        test_labels = mnist.test._labels

        return {
            "train_data": train_data,
            "train_labels": train_labels,
            "test_data": test_data,
            "test_labels": test_labels
        }