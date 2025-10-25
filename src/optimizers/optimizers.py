"""
Contains optimizer classes for use in other python scripts.
"""
import numpy as np

class ADAM_Optimizer:
    def __init__(self, learning_rate=0.01, beta_1=0.9, beta_2=0.999,
                epsilon=1e-8):
        self.learn_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = 0  # Momentum
        self.v = 0  # Adaptive learning rate
        self.t = 0  # Timestep
        
    def update(self, gradient):
        self.t += 1
        
        # Update biased first moment estimate
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        
        # Update biased second raw moment estimate  
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (gradient ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta_1 ** self.t)
        
        # Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1 - self.beta_2 ** self.t)
        
        # Update parameters
        update = self.learn_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return update
    
    def reset(self):
        self.m = 0
        self.v = 0
        self.t = 0

class SGD_Optimizer:
    """
    Stochastic Gradient Descent with optional momentum.
    """
    def __init__(self, learning_rate=1e-2, momentum=0):
        self.learn_rate = learning_rate
        self.momentum = momentum
        self.velocity = 0

    def update(self, gradient):
        self.velocity = self.momentum * self.velocity - self.learn_rate * gradient
        return self.velocity

    def reset(self):
        self.velocity = 0