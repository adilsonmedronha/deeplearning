import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

import wandb
wandb.init(project="vae")


import torchvision.transforms as transforms
from torchvision.datasets import CelebA

# transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

# celeba_path = "./data/CelebA"

# train_dataset = CelebA(root=celeba_path, split='train', transform=transform, download=True)
# val_dataset = CelebA(root=celeba_path, split='test', transform=transform, download=True)

# batch_size = 8
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
val_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
batch_size = 16
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print("Number of training examples:", len(train_dataset))
print("Number of test examples:", len(val_loader))


def viz(images_batch):
    grid = torchvision.utils.make_grid(images_batch, nrow=8, padding=2)
    grid = grid.numpy().transpose((1, 2, 0))

    plt.figure(figsize=(10, 10))
    plt.imshow(grid)
    plt.axis('off')
    plt.savefig('celeba.png')
    plt.close()

data_iterator = next(iter(train_loader))
images_batch, labels_batch = data_iterator
viz(images_batch)


def elbo_loss(x, pred, z_mean, z_logvar):
    """
    loss = reconstruction_loss + KL
    KL: KL divergence between q(z|x) and p(z)
        KL[ q(z|x) || p(z) ] <==> KL[ N(u, std) || |N(0, 1) ]
        https://arxiv.org/pdf/1312.6114.pdf...
        "Solution of -DKL(q(z)||p(z)), Gaussian case" p. 10
        KL = [sum(1 + log(sigma^2) - mu^2 - sigma^2)] / 2
    Args:
        x (_type_): _description_
        pred (_type_): _description_
        z_mean (_type_): _description_
        z_std (_type_): _description_

    Returns:
        _type_: _description_
    """
    mse_loss = nn.MSELoss(reduction='sum')
    # print(f"elbo loss    x.shape >> {x.shape}")
    # print(f"elbo loss pred.shape >> {pred.shape}")
    reconstruction_loss = mse_loss(pred, x)
    kl_divergence = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
    return reconstruction_loss + kl_divergence


class Vae(nn.Module):
    def __init__(self, latent_dim=20, input_dim=784):
        super(Vae, self).__init__()
        self.input_dim = input_dim

        # encoder <==> q(z|x)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, latent_dim * 2)
        )

        # decoder <==> p(x|z)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, self.input_dim),
            nn.Sigmoid()
        )

    def trick(self, mean, std):
        epsilon = torch.randn_like(std)
        return mean + std * epsilon

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        z = self.encoder(x)
        mu, logvar = torch.chunk(z, 2, dim=1)
        std = torch.exp(0.5 * logvar)
        reparameterized = self.trick(mu, logvar)
        pred = self.decoder(reparameterized)
        pred = pred.reshape(-1, 1, 28, 28)
        # pred = pred.reshape(-1, 3, 64, 64)
        return pred, mu, std
    

def train(model, train_loader, optimizer, device, indices_images):
    loss_sum = 0
    batch_loss = []
    num_samples = len(train_loader.dataset)
    model.train()
    images = []
    with tqdm(total=num_samples, desc="Training", unit="batch") as pbar:
        for idx, (x, _) in enumerate(train_loader):
            optimizer.zero_grad()
            x = x.to(device)
            pred, mu, std = model(x)
            loss = elbo_loss(x, pred, mu, std)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            pbar.update(x.size(0))
            # pbar.set_postfix(loss=loss_sum / ((idx + 1) * train_loader.batch_size))
            if idx in indices_images:
                pred = pred.reshape(-1, *(1, 28, 28)).detach()
                images.append(pred.cpu())
            batch_loss.append(loss.item())

    images = torch.cat(images, dim=0)
    return (loss_sum / num_samples), batch_loss, images

def val(model, val_loader, device, indices_images):
    
    val_loss = 0
    num_samples = len(val_loader.dataset)
    model.eval()
    images = []
    batch_loss = []

    with torch.no_grad(), tqdm(total=num_samples, desc="Validation", unit="batch") as pbar:
        for idx, (x, _) in enumerate(val_loader):
            x = x.to(device)
            pred, mu, std = model(x)
            loss = elbo_loss(x, pred, mu, std).item()
            val_loss += loss
            pbar.update(x.size(0))

            if idx in indices_images:
                pred = pred.reshape(-1, *(1, 28, 28)).detach()
                images.append(pred.cpu())
            batch_loss.append(loss)
        
    images = torch.cat(images, dim=0)
    return (val_loss / num_samples), batch_loss, images

def train_model(model, 
                train_loader, 
                val_loader, 
                optimizer,
                device,
                epochs=10, 
                early_stop_patience=5,
                n_images=4):
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                     mode='min', 
                                                     factor=0.1, 
                                                     patience=5, 
                                                     verbose=True)
    
    indices_images = np.linspace(0, len(train_loader), n_images, dtype=int)

    best_val_loss = float('inf')
    early_stop_counter = 0
    for epoch in range(epochs):
        train_loss, batch_train_loss, images_train = train(model, 
                                                           train_loader, 
                                                           optimizer, 
                                                           device, 
                                                           indices_images)
        
        val_loss, batch_val_loss, images_val  = val(model, 
                                                    val_loader, 
                                                    device, 
                                                    indices_images)

        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print(f'Early stopping at epoch {epoch}.')
                break
        
        wandb.log({'train_loss': train_loss, 
                   'val_loss': val_loss, 
                   'best_val_loss': best_val_loss,
                   'pred train': wandb.Image(images_train, caption='train'),
                   'pred val': wandb.Image(images_val, caption='val')},
                    step=epoch+1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# model = Vae(latent_dim=10, input_dim=64*64*3)
model = Vae(latent_dim=10, input_dim=28*28)

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
train_model(model, 
            train_loader, 
            val_loader, 
            optimizer, 
            device, 
            epochs=50,  
            early_stop_patience=5)