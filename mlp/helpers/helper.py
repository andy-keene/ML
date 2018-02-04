import json
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

def rescale_one_hots(keys, dataset):
    '''
    Rescales data in the dataset from 1 to 0.9 and 0 to 0.1
    Args:
        keys (dict): data elements to apply rescaling to
        dataset (dict): where dataset[keys] (np.array)
    '''
    for key in keys:
        dataset[key] = np.where(dataset[key] > 0, 0.9, 0.1)
    return dataset

def print_matrix_as_table(file_name):
    '''
    Prints the confusion matrix of a json obj as a pretty table
    Args:
        file_name (string): file path of .json data, must contain 'confusion_matrix' key
    Returns:
        None
    '''
    headers = [str(x) for x in range(0,10)]
    file_json = dict()

    with open(file_name, 'r') as f:
        file_json = json.load(f)
    confusion_matrix = np.array(file_json['confusion_matrix'])
    pretty_table = tabulate(confusion_matrix, headers, tablefmt="grid")
    confusion_matrix = np.array(file_json['confusion_matrix'])
    print(pretty_table)
    print('Final accuracy: ', np.trace(confusion_matrix) / np.sum(confusion_matrix))

def check_data_size(dataset):
    '''prints size of each value in the dataset'''
    for key in dataset:
        print('{} shape {}'.format(key, dataset[key].shape))

def trim_dataset_for_test(mnist_data, size):
    '''trims dataset to given size (int)'''
    for key in mnist_data:
        mnist_data[key] = mnist_data[key][:size,:]

def test_perceptron():
    '''simple test for perceptron'''
    input_data = np.random.randint(-5, 5, size=(5, 3))
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
    perceptron.train(input_data, input_labels)

def save_data(directory, file_name, results):
    '''
    Saves performance data to:
        directory/file_name.json: raw data
        directory/file_name.png: plot of accuracy over epochs
    '''    
    epochs = []
    test_accuracy = []
    train_accuracy = []

    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(file_name + '.json', 'w') as f:
        json.dump(results, f)
    for key in results['accuracy']:
        epochs.append(key)
        train_accuracy.append(results['accuracy'][key]['train'])        
        test_accuracy.append(results['accuracy'][key]['test'])

    plt.plot(epochs, train_accuracy)
    plt.plot(epochs, test_accuracy)
    plt.xlabel('epoch')    
    plt.ylabel('accuracy')
    plt.legend(['training data', 'test data'], loc='upper left')
    plt.ylim(min(train_accuracy + test_accuracy) - 0.1, max(train_accuracy + test_accuracy) + 0.1)
    plt.savefig(file_name + '.png', bbox_inches='tight')
    plt.clf()