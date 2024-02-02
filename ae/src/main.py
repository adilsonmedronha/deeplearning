import numpy as np

from nn import NeuralNetwork
from loss import MSE
from activation import ReLU, Sigmoid
from layer import Layer
from optimizer import Adam

from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


nn = NeuralNetwork(
    [Layer(784, 32, ReLU()),
      Layer(32, 16, ReLU()),
      Layer(16, 32, ReLU()),
      Layer(32, 784, Sigmoid())])

mse = MSE(nn)
optimizer = Adam(nn, lr=0.005)


transform = transforms.Compose([transforms.ToTensor()])

train_dataset = MNIST(root='data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

x, y = next(iter(train_loader))
x = x.numpy().reshape(784,-1)
print(x.shape)
pred = nn(x)

import matplotlib.pyplot as plt
import os
save_dir = 'preds'
os.makedirs(save_dir, exist_ok=True)
losses = []
iterations = []
for i in range(1000):
    pred = nn(x)
    optimizer.zero_grad()
    loss = mse(pred, x)
    mse.backward()
    optimizer.step()
    
    print(f"{i} w_0 {np.mean(abs(nn.layers[0].W))}")
    print(f"{i} b_0 {np.mean(abs(nn.layers[0].b))}")

    print(f"{i} grad_0 {np.mean(abs(nn.layers[0].grad_local))}")
    print(f"{i} delta_0 {np.mean(abs(nn.layers[0].delta))}")


    plt.subplot(1, 2, 1)
    plt.imshow(x.reshape(28, 28), cmap='gray')
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(pred.reshape(28, 28), cmap='gray')
    plt.title('Predicted Image')
    plt.savefig(f'preds/iteration_{i}_images.png')
    plt.clf()    
    plt.close()

    losses.append(loss)
    iterations.append(i)

# Plotting the loss curve
plt.figure()
plt.plot(iterations, losses, label='MSE Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('MSE Loss Over Iterations')
plt.legend()
plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
plt.show()
plt.close()

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
