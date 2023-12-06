import numpy as np 

class ActivationFunction : 
    def sigmoid(self, X, W, b) :
        return 1.0/(1.0 + np.exp(-(np.dot(W.T, X) + b)))
    
    def tanh(self, X, W, b) :
        z = np.exp(-(np.dot(W.T, X)+b))
        return (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))
    
    def relu(self, X, W, b) :
        x = np.dot(W.T, X) + b
        return np.maximum(x, 0)
    
    def softmax(self, X, W, b) :
        z_exp = np.exp(np.dot(W, X) + b)
        z_exp_sum = np.sum(z_exp)
        return z_exp/z_exp_sum
    
W = np.array([0.1, 0.2, 0.6])
X = np.array([0.2, 0.1, 0.3])
b = 1.5


def softmax(X, W, b) : 
    z = np.exp(np.dot(W, X) + b)
    return z/np.sum(z)

w = np.array([[1, 2, 3], [2, 3, 8], [1, 5, 7]])

print(softmax(X, W, b))