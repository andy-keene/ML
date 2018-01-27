import time
from perceptron import Perceptron
from helpers.dataset import Dataset
import helpers.helper as hp

def main():
    #dir of form ./save/Sat-Jan-27-13:52:05-2018
    directory = './save/{}'.format(time.ctime().replace(' ', '-'))
    file = directory + '/{}'
    epochs = 60
    etas = [0.1, 0.01, 0.001]
    mnist_data = Dataset.load()
    hp.check_data_size(mnist_data)
    
    for eta in etas:
        perceptron = Perceptron(mnist_data["train_data"].shape[1],
                                mnist_data["train_labels"].shape[1],
                                eta)
        results = perceptron.run_training(mnist_data, epochs)
        hp.save_data(directory, file.format(eta), results)

if __name__ == '__main__':
    main()
