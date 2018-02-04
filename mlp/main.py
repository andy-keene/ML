import time
from mlp import MLP
from helpers.dataset import Dataset
import helpers.helper as hp
import numpy as np

def main():
    #dir of form ./save/Sat-Jan-27-13:52:05-2018
    directory = './save/{}'.format(time.ctime().replace(' ', '-'))
    file = directory + '/{}'
    default_epochs = 50
    mnist_data = Dataset.load()
    input_dim = mnist_data["train_data"].shape[1]
    output_dim = mnist_data["train_labels"].shape[1]
    training_examples_num = mnist_data["train_data"].shape[0]
    mnist_data = hp.rescale_one_hots(['train_labels', 'test_labels'], mnist_data)
    hp.check_data_size(mnist_data)
    
    #hidden nodes tests
    for num_hiddens in [20, 50, 100]:
        mlp =  MLP(input_dim, output_dim, hidden_dim=num_hiddens)
        results = mlp.run_training(mnist_data, default_epochs)
        hp.save_data(directory, file.format(str(num_hiddens) + '-hiddens'), results)
    
    #momentum tests
    for momentum in [0, 0.25, 0.5]:
        mlp =  MLP(input_dim, output_dim, momentum=momentum)
        results = mlp.run_training(mnist_data, default_epochs)
        hp.save_data(directory, file.format(str(momentum) + '-momentum'), results)

    #vary number of inputs
    _all_train_data = mnist_data["train_data"]
    _all_train_labels = mnist_data["train_labels"]
    for percent in [0.25, 0.5]:
        #reduce training set by some % (input order is already randomized)
        reduction = int(training_examples_num * percent)
        mnist_data['train_data'] = _all_train_data[:reduction,:]
        mnist_data['train_labels'] = _all_train_labels[:reduction,:]        
        mlp =  MLP(input_dim, output_dim)
        results = mlp.run_training(mnist_data, default_epochs)
        hp.save_data(directory, file.format(str(percent) + '-percent-inputs'), results)

if __name__ == '__main__':
    main()
