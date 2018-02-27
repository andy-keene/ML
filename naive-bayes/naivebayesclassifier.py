import numpy as np

class NaiveBayesClassifier(object):
    '''Classifies feature vectors using Naive Bayes'''

    def __init__(self, class_values=[0,1]):
        self.classes = {value : dict() for value in class_values}
        self.num_classes = len(class_values)
    
    def learn(self, train_data):
        '''
        Args:
            train_data (np.array): training set of which to learn learn prior, std, and mean
        '''

        for _class in self.classes:
            cond = np.where(train_data[:,-1] == _class)
            datums =  train_data[cond][:,:-1]
            mean = datums.mean(axis=0)
            std = datums.std(axis=0)
            #avoid 0 standard deviations
            std = np.where(std == 0.0, 0.00001, std)
            self.classes[_class] = {
                'prior': datums.shape[0] / train_data.shape[0],
                'std': std,
                'mean': mean
            }

    def classify(self, data):
        '''
        Args:
            data (np.array): of shape (m,n) where there are m documents each with n - 1
                             features, with the n-th column being the class label
        Returns:
            np.array: A confusion matrix of shape (number of class, number of classes) 
        '''
        confusion_matrix = np.zeros((self.num_classes, self.num_classes))

        for datum in data:
            features = datum[:-1]
            label = int(datum[-1])
            posteriors =  {}

            for _class, _class_values in self.classes.items():
                _posterior = self._get_posterior(features, _class_values)
                posteriors[_class] = _posterior

            #get the predicted label
            predicted = max(posteriors.items(), key=lambda n: n[1])[0]
            confusion_matrix[predicted][label] += 1
        return confusion_matrix

    def _get_posterior(self, features, class_values):
        ''' Returns the naive bayes approx. of the posterior P(class | features) 
        Args:
            features (np.array): of shape (1, m) where m is the number of features
            class_values (dict): with mean=nnp

        Returns:
            float: Posterior of the class, c, given the features
                   i.e. returns log(P(c) + log(P(x-1 | class)) + ... + log(P(x-n | c))
        '''
        posterior = np.log(class_values['prior'])
        for i in range(features.shape[0]):
            posterior += np.log(self._gaussian(features[i], class_values['mean'][i], class_values['std'][i]))
        return posterior
    
    def _gaussian(self, x, mu, sigma):
        ''' Returns the gaussian approx. of the class conditional probability
        Args:
            x (float): a single feature, component i, of a feature vector
            mu (float): mean of the i-th component
            sigma (float): standard deviation of i-th component
        Returns:
            float: P(x | class)
        '''
        u = ((x - mu)**2 / (2*(sigma)**2))
        result = np.exp(-u) / (np.sqrt(2*np.pi) * sigma)
        if result <=0:
            pass
        #check if 0
        return result if result != 0 else 0.00000000000001
