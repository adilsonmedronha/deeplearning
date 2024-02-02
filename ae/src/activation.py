from abc import ABC, abstractmethod
import numpy as np

class Activations(ABC):
    @abstractmethod
    def __call__(self, x):
        pass 

    @abstractmethod
    def compute_grad(self):
        pass

class ReLU(Activations):
    def __call__(self, x):
        self.activated = np.maximum(0, x)
        return self.activated
    
    def compute_grad(self):
        # TODO lancar um erro caso chamarem compute_grad antes de a(z)
        return np.where(self.activated > 0, 1, 0)

class Sigmoid(Activations):
    def __call__(self, x):
        # self.activated = np.clip(1 / (1 + np.exp(-x)), 0.01, 0.99)
        self.activated = 1 / (1 + np.exp(-x))
        return self.activated
    
    def compute_grad(self):
        # TODO lancar um erro caso chamarem compute_grad antes de a(z)
        return self.activated * (1 - self.activated)
    

# class LeakyReLU(Activations):
#     def __call__(self, x):
#         self.activated = 
