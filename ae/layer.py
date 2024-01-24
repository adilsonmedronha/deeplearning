import numpy as np
from activation import Activations


class Layer():
    def __init__(self, input_size: int, output_size: int, activation: Activations):
        
        #self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / (input_size + output_size)) 
        #self.b = np.random.randn(output_size, 1) * np.sqrt(2.0 / (input_size + output_size)) 

        self.W = np.random.randn(input_size, output_size)  
        self.b = np.random.randn(output_size, 1)  
        self.z, self.a, self.x = 0, 0, 0
        self.grad_local, self.delta = 0, 0
        self.activation = activation

    def __call__(self, x):
        self.x = x
        self.z = self.W.T @ x + self.b
        self.a = self.activation(self.z)
        return self.a
    