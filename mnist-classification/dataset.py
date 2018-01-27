from tensorflow.examples.tutorials.mnist import input_data
from pathlib import Path
import numpy as np

class Dataset(object):
    '''
    Handler for mnist data set
    '''
    file_archive = './MNIST_data/mnist_numpy.npz'

    def load():
        '''
        Loads mnist data from .npz, downloads and saves an archive if it does not exist
        *needs ./MNIST_data/ dir
        '''
        mnist_data = dict()

        if Path(Dataset.file_archive).is_file():            
            with np.load(Dataset.file_archive) as archive:
                mnist_data = {
                    "train_data": archive['train_data'],
                    "train_labels": archive['train_labels'],
                    "test_data": archive['test_data'],
                    "test_labels": archive['test_labels']
                }
        else:
            mnist_data = Dataset.save_mnist_from_tf()
        return mnist_data

    def save_mnist_from_tf():
        '''
        returns dict:
            train_data (np.array):  of shape (60000, 784)
            train_labels (np.array): of shape (60000, 10)
            test_data (np.array): of shape (10000, 784)
            test_labels (np.array): of shape (10000, 10)
        '''
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        train_data = np.vstack((mnist.train._images, mnist.validation._images))
        train_labels = np.vstack((mnist.train._labels, mnist.validation._labels))
        test_data = mnist.test._images
        test_labels = mnist.test._labels

        np.savez_compressed = np.savez_compressed(Dataset.file_archive,
                            train_data=train_data,
                            train_labels=train_labels,
                            test_data=test_data,
                            test_labels=test_labels)

        return {
            "train_data": train_data,
            "train_labels": train_labels,
            "test_data": test_data,
            "test_labels": test_labels
        }