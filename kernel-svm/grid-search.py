import json
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from preprocessor import Preprocessor
from graph import plot_svm

features = [
            "Name",
            "Sex",
            "SibSp",
            "Fare"
]

results_file = './grid-search.csv'
train_file = './data/titanic_train.csv'
preprocessor = Preprocessor(train_file)
dataset = preprocessor.get_matrix(features + ['Survived'])
data, labels = dataset[:,:-1], dataset[:,-1]


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
        3
    ]
}
svc = SVC(cache_size=2000)
clf = GridSearchCV(svc, param_grid=parameters, cv=8, return_train_score=True)
results = clf.fit(data, labels)
pd.DataFrame(results.cv_results_).to_csv(results_file)


