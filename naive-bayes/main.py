from naivebayesclassifier import NaiveBayesClassifier
import numpy as np

def load_data(filename): 
    '''
    Loads numpy data
    Returns:
        dict: {'train_data': (np.array), 'test_data': (np.array)}    
    '''
    data = np.loadtxt(filename, delimiter=',', dtype=np.float64)

    #shuffle and split
    np.random.shuffle(data)
    split = data.shape[0] // 2
    train_data = data[:split,:]
    test_data = data[split:,]

    return {
        'train_data': train_data,
        'test_data': test_data
    }

def main():
    #this program style is getting kind of boring...
    data = load_data('./spambase/spambase.data')

    classifier = NaiveBayesClassifier(class_values=[0,1])
    classifier.learn(data['train_data'])
    confusion_matrix = classifier.classify(data['test_data'])

    print('confustion matrix:\n{}\naccuracy: {} '.format(confusion_matrix, 
        np.trace(confusion_matrix) / np.sum(confusion_matrix)
        ))

if __name__ == '__main__':
    main()