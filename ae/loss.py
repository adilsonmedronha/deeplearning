import numpy as np
from abc import ABC, abstractmethod
from nn import NeuralNetwork
from typing import Union, List
from layer import Layer


class Loss(ABC):
    def __init__(self, model: NeuralNetwork):
        self.model = model

    @abstractmethod
    def __call__(sefl, input: Union[np.ndarray, list], pred: Union[np.ndarray, list]):
        pass

    @abstractmethod
    def dloss(self):
        pass

    def compute_delta(self, layers: List[Layer], l: int):
        if l == len(layers) - 1:
            return self.dloss() * layers[l].activation.compute_grad()
        else:
            return (layers[l + 1].W @ layers[l + 1].delta) * layers[l].activation.compute_grad()

    def compute_grad_local(self, layers: List[Layer], l: int):
        if l == 0:
            return layers[l].delta @ layers[l].x.T
        else:
            return layers[l].delta @ layers[l - 1].a.T

    def backward(self):
        layers = self.model.get_layers()
        L = len(layers) 

        for l in range(L-1, -1, -1):
            layers[l].delta = self.compute_delta(layers, l)
            layers[l].grad_local = self.compute_grad_local(layers, l)

    def __repr__(self) -> str:
        info = [f'layer {l} delta: {layer.delta.shape} local {layer.grad_local.shape}\n' for l,
                layer in enumerate(self.model.layers)]
        return f"{self.__class__.__name__}(\n{''.join(info)})"


class MSE(Loss):
    def __call__(self, pred: Union[np.ndarray, list], y: Union[np.ndarray, list]):
        self.error = pred - y
        _, self.m = pred.shape
        return np.sum(self.error @ self.error.T) / self.m

    def dloss(self):
        return np.sum(self.error) / self.m
    

class BCE(Loss):
    def __call__(self, pred: Union[np.ndarray, list], y: Union[np.ndarray, list]):
        y*np.log(pred) + (1 - y)*np.log(1 - pred)

    def dloss(self):
        pass