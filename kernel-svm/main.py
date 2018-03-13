import json
import time
import os
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from preprocessor import Preprocessor
from graph import plot_svm

def save_data(directory, file_name, results, file_type='.json'):
    '''
    Saves performance data to:
        directory/file_name: raw data
        results (dict | pd.dataframe):
        file_type (string): .json or .csv
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    if file_type == '.json':
        with open(file_name + file_type, 'w') as f:
            json.dump(results, f)
    elif file_type == '.csv':
        results.to_csv(file_name)

def main():
    #files
    directory = './save/{}'.format(time.ctime().replace(' ', '-'))
    model_file = directory + '/model-set-{}'
    results_file = directory + '/results-set-{}'
    train_file = './data/titanic_train.csv'

    # set grid search for hyper params
    cv_size = 4
    cache_size = 4000
    max_iterations = 100000
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
            'rbf',
            'linear',
            'poly'
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
            3,
            5
        ]
    }
    preprocessor = Preprocessor(train_file)

    for set_num, features in enumerate(feature_set):
        dataset = preprocessor.get_matrix(features + ['Survived'])
        data, labels = dataset[:,:-1], dataset[:,-1]
        svc = SVC(cache_size=cache_size, max_iter=max_iterations)
        clf = GridSearchCV(svc, param_grid=parameters, cv=cv_size, return_train_score=True)
        results = clf.fit(data, labels)
        
        #save results
        top_model = clf.get_params()
        top_model['training_feature_set'] = features
        top_model['estimator'] = str(top_model['estimator'])
        save_data(directory,
            results_file.format(set_num),
            pd.DataFrame(results.cv_results_),
            file_type='.csv'
        )
        save_data(directory,
            model_file.format(set_num),
            top_model,
            file_type='.json'
        )


if __name__ == '__main__':
    main()