import time
from mlp import MLP
from helpers.dataset import Dataset
import helpers.helper as hp
import numpy as np

def main():
    #dir of form ./save/Sat-Jan-27-13:52:05-2018
    directory = './save/{}'.format(time.ctime().replace(' ', '-'))
    file = directory + '/{}'
    epochs = 60
    hidden_nodes = 20
    etas = [0.1, 0.01, 0.001]
    mnist_data = Dataset.load()
    hp.check_data_size(mnist_data)
    
    mnist_data['train_labels'] = np.where(mnist_data['train_labels'] > 0, 0.9, 0.1)
    mnist_data['test_labels'] = np.where(mnist_data['test_labels'] > 0, 0.9, 0.1)
    
    print('hello')
    mlp = MLP(mnist_data["train_data"].shape[1],
        hidden_nodes,
        mnist_data["train_labels"].shape[1])
    print('hello')
    
    mat = mlp.test(mnist_data['test_data'], mnist_data['test_labels'])
    accuracy = np.trace(mat) / np.sum(mat)
    for n in range(300):
        print('train ', n)
        mlp.train(mnist_data['train_data'], mnist_data['train_labels'])
        
    mat = mlp.test(mnist_data['test_data'], mnist_data['test_labels'])
    print('accuracy delta: ', (np.trace(mat) / np.sum(mat)) - accuracy)
    print('raw accuracy ', (np.trace(mat) / np.sum(mat)))
    #results = perceptron.run_training(mnist_data, epochs)
    #hp.save_data(directory, file.format(eta), results)

if __name__ == '__main__':
    main()


'''
def test():
    anddata = np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,1]])
    xordata = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])
    
    
    mlp = MLP(2, 2, 1, 3, 1, 2, 1)

    for x, y in mlp.minibatch(anddata[:,0:2], anddata[:,-1:]):
        print('input batch = {} \noutput batch = {}\n'.format(x,y))
'''