import pandas as pd
import numpy as np

class Preprocessor(object):
    def __init__(self, filename):
        self.df = pd.read_csv(filename)
        pass

    def get_matrix(self, cols):
        '''
        Args:
            cols (list): subset of the dataframe to retrieve
        Returns:
            np.array: with columns matching cols
        '''
        _data = self.df[cols].as_matrix()
        _data = np.nan_to_num(_data)
        return _data

    def read_file(filename):
        pass

    def get_df(self):
        ''' Returns a copy of the dataframe'''
        return self.df.copy(deep=True)

    def embarked_to_int(embarked):
        if embarked == 'S':
            return 0
        elif embarked == 'Q':
            return 1
        elif embarked == 'C':
            return 2
        else:
            return 3