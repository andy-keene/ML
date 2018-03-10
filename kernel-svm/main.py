import json
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from preprocessor import Preprocessor
from graph import plot_svm

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
column_options = [
    'Pclass',
    'Name',
    'Sex',
    'Age',
    'SibSp',
    'Parch',
    'Ticket',
    'Fare',
    'Cabin',
    'Embarked'
]
feature_combinations = [
    ['Pclass', 'Name', 'Sex'],
    ['Pclass', 'Name', 'Sex', 'SibSp', 'Fare'],
    ['Sex', 'SibSp', 'Parch', 'Fare'],    
]
kernel_types = ['rbf', 'linear', 'poly']
hyperparams = [
    (0.2, 1),
    (3, 3),
    (8, 5),
    (15, 5),
]
k_folds = 8
cache_size = 7000
preprocessor = Preprocessor(train_file)
model_runs = {}
trial = 0

#kernel/model selection
for features in feature_combinations:
    #feature selection
    for kernel in kernel_types:
        #param testing
        for c, degree in hyperparams:
            dataset = preprocessor.get_matrix(features + ['Survived'])
            data, labels = dataset[:,:-1], dataset[:,-1]
            svm = SVC(kernel=kernel, C=c, cache_size=cache_size)
            scores = cross_val_score(svm, data, labels, cv=k_folds)
            model_runs[trial] = {
                'kernel': kernel,
                'features': features,
                'C': c,
                'Degree': degree,
                'Scores': list(scores),
                'std': float(scores.std()),
                'mean': float(scores.mean())
            }
            print('tested {}'.format(model_runs[trial]))
            trial += 1

#save
print('saving data')
with open('./model-runs.json', 'w') as f:
    json.dump(model_runs, f)



        

'''
#data = preprocessor.get_matrix(['PassengerId', 'Sex', 'Ticket', 'Pclass', 'Fare', 'Survived'])
dataset = preprocessor.get_matrix(['Age', 'Fare', 'Survived'])

data, labels = dataset[:,:-1], dataset[:,-1]

print(data[:10,:])

svm = SVC(kernel='rbf', C=15)
scores = cross_val_score(svm, data, labels, cv=8)
svm.fit(data, labels)
print('Accuracy: {} +/- {}'.format(scores.mean(), scores.std()))
plot_svm(svm, './yo', 'Pclass', 'Fare', data[:100], labels[:100])


resp = svm.fit(train, train_labels)
print(resp)
predictions = svm.predict(test)
print(predictions - test_labels)
err = np.sum(abs(test_labels - predictions))
print('err = ', err / test_labels.shape[0])
'''