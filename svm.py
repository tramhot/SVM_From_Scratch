import numpy as np

class SupportVectorMachine():
    def __init__(self, 
                 learning_rate = 0.001, 
                 lambda_param = 0.01 , 
                 n_tiers=10000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.iteration=n_tiers
        self.w = None
        self.b = None
        
    def _compute_gradient_descent():
        pass
        
    def fit(self, X, y):
        y_ = np.where(y>0, 1, -1)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0 
        for _ in range(self.iteration):
            for idx, x_i in enumerate(X):
                condition =  y_[idx]*(np.dot(x_i, self.w)-self.b)>=1
                if condition:
                    self.w -= self.lr*(2*self.lambda_param*self.w)
                else:
                    self.w -= self.lr*(2*self.lambda_param*self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr*y_[idx]
        
    
    def predict(self, X):
        output = np.dot(X, self.w) - self.b
        return np.sign(output)