import json
import time
import os
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.decomposition import PCA
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
        results.to_csv(file_name, index=False)

def get_predictions(model, features, test_file, pca=None):
    '''
    Returns:
        pd.dataframe: predictions with cols ['PassengerId', 'Survived']
    '''
    preprocessor = Preprocessor(test_file)
    data = preprocessor.get_matrix_scaled(features)
    
    if pca:
        data = pca.transform(data)
    
    predictions = model.predict(data)
    df = preprocessor.get_dataframe()
    return pd.DataFrame(data={'PassengerId': df['PassengerId'], 'Survived': predictions.astype(int).tolist()})

def main():
    #files
    directory = './save/{}'.format(time.ctime().replace(' ', '-'))
    model_file = directory + '/model-set-{}'
    results_file = directory + '/results-set-{}'
    predictions_file = directory + '/predictions-set-{}'
    train_file = './data/titanic_train.csv'
    test_file = './data/titanic_test.csv'    

    #data preprocessing
    use_pca = True
    pca_n_components = 4
    preprocessor = Preprocessor(train_file)
    pca = PCA(n_components=pca_n_components)

    # set grid search for hyper params
    cv_size = 4
    cache_size = 5000
    max_iterations = 1000000

    feature_set = [
        ['Pclass', 'Sex', 'Age', 'Fare'],
        ['Name', 'Sex', 'SibSp', 'Fare'],
        ['Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Fare'],
        ['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],
        ['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin'],
        ['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare'],        
    ]
    original_feature_set = [
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
        #    'rbf',
            'linear',
            'poly'
        ],
        'C': [
            0.1,
            0.8,
            1,
            10
        ],
        'gamma': [
            1,
            3,
            4,
            5,
            7,
            10
        ],
        'degree': [
            1,
            3,
            4,
            5
        ]
    }

    for set_num, features in enumerate(feature_set):
        #dataset = preprocessor.get_matrix_scaled(features + ['Survived'])
        #data, labels = dataset[:,:-1], dataset[:,-1]
        data, labels = preprocessor.get_matrix_scaled(features), preprocessor.get_labels()

        if use_pca:
            data = pca.fit_transform(data)
        
        svc = SVC(cache_size=cache_size, max_iter=max_iterations)
        clf = GridSearchCV(svc, param_grid=parameters, cv=cv_size, return_train_score=True)
        results = clf.fit(data, labels)
        
        #test best model, and save predictions for submission
        top_model = clf.get_params()
        top_svc = top_model['estimator']
        top_svc.fit(data, labels)
        save_data(directory,
            predictions_file.format(set_num),
            get_predictions(top_svc, features, test_file, pca=pca),
            file_type='.csv'
        )

        #save results
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