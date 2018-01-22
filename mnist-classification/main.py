import numpy as np
from perceptron import Perceptron
from dataset import Dataset

def main():
    mnist_data = Dataset.load_mnist()
    for key in mnist_data:
        print('{} has shape {}'.format(key, mnist_data[key].shape))
    pass

if __name__ == '__main__':
    main()
