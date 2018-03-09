import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None

class Preprocessor(object):
    '''Reads and preprocesses the Titanic Dataset'''

    def __init__(self, filename):
        self.preprocessing_map = {
            'Name': self._name,
            'Sex': self._sex,
            'Embarked': self._embarked,
            'Ticket': self._zero,
            'Cabin': self._zero
            # ticket and cabin do not have preprocesing funcs
        }
        self.dataset_df = pd.read_csv(filename)
        self.processed_df = self._preprocess(self.dataset_df)

    def get_matrix(self, cols):
        '''
        Args:
            cols (list): list of dataframe columns to retrieve
        Returns:
            np.array: preprocessed data where np.array[:,n] == col[n]
        '''
        return np.nan_to_num(self.processed_df[cols].as_matrix())

    def get_dataframe(self):
        '''
        Returns:
            pd.dataframe: a copy of the original dataset
        '''
        return self.dataset_df.copy(deep=True)

    def _preprocess(self, original_df):
        '''
        Args:
            original_df (pd.dataframe): dataframe
        Returns:
            pd.dataframe: a preprocessed copy of original_df according to self.preprocessing_map rules
        '''
        _processed_df = original_df.copy(deep=True)
        for field in self.preprocessing_map.keys():
            _processed_df[field] = _processed_df[field].apply(self.preprocessing_map[field])
        return _processed_df

    # preprocessing functions
    # TODO: move instance methods -> external | static ?...
    def _sex(self, sex):
        ''' passanger.sex -> (int)'''
        if sex == 'female':
            return 1
        elif sex == 'male':
            return 0
        else:
            return -1

    def _embarked(self, embarked):
        ''' passanger.embarked -> (int)'''         
        if embarked == 'S':
            return 0
        elif embarked == 'Q':
            return 1
        elif embarked == 'C':
            return 2
        else:
            return 3
    
    def _name(self, name):
        ''' passanger.name -> (int)'''                 
        if 'Sir.' in name:
            return 5
        elif 'Dr.' in name:
            return 4
        elif 'Mr.' in name:
            return 3
        elif 'Mrs.' in name:
            return 2
        elif 'Miss' in name:
            return 1
        else:
            return 0
    
    def _zero(self, item):
        return 0