import numpy as np

from layer import Layer
from typing import List, Union


class NeuralNetwork():
    def __init__(self, layers: List[Layer]):
        self.layers = layers

    def __call__(self, x: Union[np.ndarray, list]):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def __repr__(self): 
        info = [f'   Layer {l}: {layer.W.shape} \n' for l, layer in enumerate(self.layers)]
        return f"{self.__class__.__name__}(\n{''.join(info)})"
    
    def get_layers(self):
        return self.layers