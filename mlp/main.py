import time
from mlp import MLP
from helpers.dataset import Dataset
import helpers.helper as hp
import numpy as np

def main():
    #dir of form ./save/Sat-Jan-27-13:52:05-2018
    directory = './save/{}'.format(time.ctime().replace(' ', '-'))
    file = directory + '/{}'
    epochs = 100
    hidden_nodes = 20
    mnist_data = Dataset.load()
    mnist_data = hp.rescale_one_hots(['train_labels', 'test_labels'], mnist_data)
    hp.check_data_size(mnist_data)
    

    for epochs in [100, 200]:
        mlp =  MLP(mnist_data["train_data"].shape[1], hidden_nodes, mnist_data["train_labels"].shape[1])
        results = mlp.run_training(mnist_data, epochs)
        hp.save_data(directory, file.format(str(epochs) + '-epochs'), results)
        

    #hidden nodes tests
    #momentum tests
    #other test?

if __name__ == '__main__':
    main()


'''
GARBAGE BIN: 
mat = mlp.test(mnist_data['test_data'], mnist_data['test_labels'])
    accuracy = np.trace(mat) / np.sum(mat)
    
    for n in range(300):
        print('train ', n)
        mlp.train(mnist_data['train_data'], mnist_data['train_labels'])
        
    mat = mlp.test(mnist_data['test_data'], mnist_data['test_labels'])
    print('accuracy delta: ', (np.trace(mat) / np.sum(mat)) - accuracy)
    print('raw accuracy ', (np.trace(mat) / np.sum(mat)))

def test():
    anddata = np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,1]])
    xordata = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])
    
    
    mlp = MLP(2, 2, 1, 3, 1, 2, 1)
'''