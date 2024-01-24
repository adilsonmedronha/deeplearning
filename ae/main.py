import numpy as np
from nn import NeuralNetwork
from loss import MSE
from activation import ReLU
from layer import Layer

from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


x = np.random.random((784, 1))

layers = [Layer(784, 4, ReLU()),
          Layer(4, 5, ReLU()),
          Layer(5, 6, ReLU()),
          Layer(6, 7, ReLU()),
          Layer(7, 2, ReLU())]

nn = NeuralNetwork(layers)
mse = MSE(nn)
y = np.random.random((2, 1))

pred = nn(x)
error = mse(pred, y)

mse.backward()




transform = transforms.Compose([transforms.ToTensor()])

train_dataset = MNIST(root='data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)



for l, layer in enumerate(nn.layers):
    print(f'l= {l}')
    print(f'x shape {layer.x.shape}')
    print(f'W shape {layer.W.shape}')
    print(f'W.T @, x.shape {(layer.W.T.shape, layer.x.shape)} = {(layer.W.T @ layer.x).shape}')
    print(f'B shape {layer.b.shape}')
    print(f'a shape {layer.a.shape}')
    print(f'z shape {layer.z.shape}')
    print(f'GRADS')
    print(f'local {layer.grad_local.shape}')
    print(f'delta {layer.delta.shape}\n')
