from sklearn import datasets
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from preprocessor import Preprocessor

train_file = './data/titanic_train.csv'
test_file = './data/titanic_test.csv'
full_file = './data/titanic_full.csv'

columns = [
    'PassengerId',
    'Pclass',
    'Name',
    'Sex',
    'Age',
    'SibSp',
    'Parch',
    'Ticket',
    'Fare',
    'Cabin',
    'Embarked',
    'Survived'
]

data_preprocessor = Preprocessor(train_file)
data = data_preprocessor.get_matrix(['Pclass', 'Fare', 'Survived'])


TEST_SIZE = 700
train, train_labels = data[:TEST_SIZE,:-1], data[:TEST_SIZE,-1]
test, test_labels = data[TEST_SIZE:,:-1], data[TEST_SIZE:,-1]

svm = SVC(kernel='rbf', C=15)
resp = svm.fit(train, train_labels)
print(resp)

predictions = svm.predict(test)
print(predictions - test_labels)
err = np.sum(abs(test_labels - predictions))
print('err = ', err / test_labels.shape[0])
