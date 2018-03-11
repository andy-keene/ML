import json
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from preprocessor import Preprocessor
from graph import plot_svm

# set grid search for hyper params
cv_size = 4
cache_size = 4000
results_file = './feature-set-{}-results.csv'
train_file = './data/titanic_train.csv'
feature_set = [
    ['Pclass', 'Sex', 'Age', 'Fare'],
    ['Name', 'Sex', 'SibSp', 'Fare'],
    ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
    ['Age', 'Sex'],
    ['Pclass', 'Fare'],
    ['Age', 'Fare'],
    ['Sex', 'Fare'],
    ['Pclass', 'Sex']
]
parameters = {
    'kernel': [
        'rbf'
    ],
    'C': [
        0.1,
        1,
        10
    ],
    'gamma': [
        1,
        3,
        5
    ],
    'degree': [
        1,
        3
    ]
}
preprocessor = Preprocessor(train_file)

for set_num, features in enumerate(feature_set):
    dataset = preprocessor.get_matrix(features + ['Survived'])
    data, labels = dataset[:,:-1], dataset[:,-1]
    svc = SVC(cache_size=cache_size)
    clf = GridSearchCV(svc, param_grid=parameters, cv=cv_size, return_train_score=True)
    results = clf.fit(data, labels)
    #save results to file
    pd.DataFrame(results.cv_results_).to_csv(results_file.format(set_num))
