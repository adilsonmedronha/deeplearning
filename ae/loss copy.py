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
            r = (self.dloss() * layers[l].activation.compute_grad())
            #print(f'dj * da 4 L{l}      = {self.dloss().shape} * {layers[l].activation.compute_grad().shape})')
            #print(f'delta global 4 L{l} = {r.shape}) ')
            return self.dloss() * layers[l].activation.compute_grad()
        else:
            r = ((layers[l + 1].W @ layers[l + 1].delta) * layers[l].activation.compute_grad())
            #print(f'(w_t+1 @ delta_l+1) * da_l = ({layers[l + 1].W.shape} @ {layers[l + 1].delta.shape}) * {layers[l].activation.compute_grad().shape}')
            #print(f'delta global l{l}          = {r.shape}')
            return (layers[l + 1].W @ layers[l + 1].delta) * layers[l].activation.compute_grad()

    def compute_grad_local(self, layers: List[Layer], l: int):
        if l == 0:
            r = (layers[l].delta @ layers[l].x.T)
            #print(f'delta_l @ x.T      = {layers[l].delta.shape} @ {layers[l].x.T.shape}')
            #print(f'delta local 0 l{l} = {r.shape})')
            return layers[l].delta @ layers[l].x.T
        else:
            r = (layers[l].delta @ layers[l - 1].a.T)
            #print(f'delta_l @ a_l-1.T      = {layers[l].delta.shape} @ {layers[l - 1].a.T.shape}')
            #print(f'delta local l{l}   = {r.shape})')
            return layers[l].delta @ layers[l - 1].a.T

    def backward(self):
        layers = self.model.get_layers()
        L = len(layers) - 1

        for l in range(L, -1, -1):
            layers[l].delta = self.compute_delta(layers, l)
            layers[l].grad_local = self.compute_grad_local(layers, l)
            print('')
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