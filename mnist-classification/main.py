import numpy as np
#import matplotlib.pyplot as plt
from perceptron import Perceptron
from dataset import Dataset

def test_perceptron():
    #2 training samples, length 3
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

def plot(epoch_accuracy):
    #garaunteed correspondence
    #https://docs.python.org/3/library/stdtypes.html#dictionary-view-objects
    epochs, accuracy = epoch_accuracy.keys(), epoch_accuracy.values()
    plt.plot(epochs, accuracy)
    plt.axis([0, max(epochs), 0, 1])
    plt.show

def main():
    epochs = 60
    etas = [0.1, 0.01, 0.001]
    mnist_data = Dataset.load_mnist()
    perceptron = Perceptron(mnist_data["train_data"].shape[0],
                            mnist_data["train_data"].shape[0],
                            0.001)
    perceptron = Perceptron(784, 10, 0.001)
    print('training')
    results = perceptron.train(mnist_data["train_data"][:50], mnist_data["train_labels"][:50])
    #plot(epoch_accuracy)
    #test_confusion_matrix = perceptron.test(mnist_data["test_data"], mnist_data["test_labels"])

    for epoch in results['accuracy']:
        print('Epoch {}: %{}'.format(epoch, results[epoch]))
    print(results['confusion_matrix'])

if __name__ == '__main__':
    main()
