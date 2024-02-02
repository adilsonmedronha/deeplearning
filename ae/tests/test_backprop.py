import unittest 
import numpy as np

from src.layer import Layer
from src.nn import NeuralNetwork
from src.activation import ReLU
from src.loss import MSE


class TestBackprop(unittest.TestCase):
    def setUp(self):
        layers = [Layer(1, 4, 
                  ReLU()), 
                  Layer(4, 5, 
                  ReLU()),
                  Layer(5, 6, 
                  ReLU()),
                  Layer(6, 7, 
                  ReLU()),
                  Layer(7, 2, 
                  ReLU())]

        self.nn = NeuralNetwork(layers)
        self.loss = MSE(self.nn)
    
    def test_call(self):
        n_feature, n_batch = 1, 8
        x = np.random.randn(n_feature, n_batch) 
        pred = self.nn(x)
        l2 = self.loss(pred, x)
        self.loss.backward()
        L = 4 # Last layer == n_layers - 1
        
        assert self.nn.layers[L].delta.shape == (2, 8)
        # layers[l].delta @ layers[l - 1].a.T => (2, 8) * (7, 8).T
        assert self.nn.layers[L-1].a.shape == (7, 8)
        assert self.nn.layers[L].grad_local.shape == (2, 7)

        # delta_3 = (w_4.T    @ delta_4) * da_3
        # delta_3 = ((7, 2)   @ (2,8)  ) * (7, 8)
        assert self.nn.layers[4].W.shape == (7, 2) 
        assert self.nn.layers[3].delta.shape == (7, 8)
        assert self.nn.layers[3].grad_local.shape == (7, 6)

        assert self.nn.layers[2].delta.shape == (6, 8)
        assert self.nn.layers[2].grad_local.shape == (6, 5)

        # assert self.nn.layers[L - 3].delta.shape == (6, 8)
        # assert self.nn.layers[L - 3].grad_local.shape == (6, 5)
