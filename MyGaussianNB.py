import numpy as np

class MyGaussianNB(object):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        self.y_probs = {}
        self.x_params = {}
        
        for class_idx in set(y):
            """firstly, calculate the prior distribution, p(y1), p(y2) ..."""
            self.y_probs[class_idx] = (y == class_idx).mean() 
            """ eg. y_probs = { 0：0.1， 1：0.2， 2：0.7}""" 
                       
            """split the train dataset"""
            mu = X[y==class_idx].mean(axis=0)
            delta = X[y==class_idx].std(axis=0)
            self.x_params[class_idx] = {"mu":mu, "delta": delta}       
    def predict(self, X):
        X = np.array(X)
        if X.ndim != 2:
            raise Exception(" it must be 2-dimensions")
                                        
        results = []
        for x in X:
            labels = []
            for label_idx in self.y_probs:
                prob = self.y_probs[label_idx]
                for feature_idx in range(len(x)):
                    prob *= self._normal_distribution(mu=self.x_params[label_idx]["mu"][feature_idx], 
                                                      delta=self.x_params[label_idx]["delta"][feature_idx],
                                                      x=x[feature_idx])
                labels.append(prob)
            results.append(labels)
        return np.array(results).argmax(axis=1)
    
    def score(self, X, y):
        y_pred = self.predict(X=X)
        return (y == y_pred).mean()
    
    def _normal_distribution(self,mu,delta,x):
        return 1 / np.sqrt(2 * np.pi) / delta * np.exp(- (x - mu) ** 2 / 2 / delta ** 2)
