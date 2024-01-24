from abc import ABC, abstractmethod
from nn import NeuralNetwork

class Optimizer(ABC):
    def __init__(self, model: NeuralNetwork, lr: float):
        self.model = model
        self.lr = lr

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def zero_grad(self):
        pass

class Adam(Optimizer):
    
    def step(self):
        layers = self.model.get_layers()
        for layer in layers:
            # TODO investigar esses transpostos, tentar deixar o backprop igual as minhas anotacoes?
            layer.W -= self.lr * layer.grad_local.T
            layer.b -= self.lr * layer.delta

    def zero_grad(self):
        layers = self.model.get_layers()
        for layer in layers:
            layer.delta = 0
            layer.grad_local = 0