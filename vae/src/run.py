import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import CelebA, MNIST
from tqdm import tqdm

import sys; sys.path.append("../utils")
from utils import create_gif_from_folder, viz, plot_losses, save_model, get_args

import wandb
wandb.init(project="vae")

def get_data_loaders(args):
    # ignore #channels
    temp_img_shape = args.img_shape[1:]
    transform = transforms.Compose([transforms.Resize(tuple(temp_img_shape)), transforms.ToTensor()])
    if args.dataset_name == "MNIST":
        train_dataset = MNIST(root=args.dataset_path, train=True, transform=transform, download=True)
        val_dataset = MNIST(root=args.dataset_path, train=False, transform=transform, download=True)
    elif args.dataset_name == "CelebA":
        train_dataset = CelebA(root=args.dataset_path, split='train', transform=transform, download=True)
        val_dataset = CelebA(root=args.dataset_path, split='test', transform=transform, download=True)
    else: raise ValueError("Dataset name not found.")
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)
    print("Number of training examples:", len(train_dataset))
    print("Number of test examples:", len(val_loader))
    return train_loader, val_loader


def elbo_loss(x, pred, z_mean, z_logvar):
    """
    loss = reconstruction_loss + KL
    KL: KL divergence between q(z|x) and p(z)
        KL[ q(z|x) || p(z) ] <==> KL[ N(u, std) || N(0, 1) ]
        https://arxiv.org/pdf/1312.6114.pdf
        "Solution of -DKL(q(z)||p(z)), Gaussian case" p. 10
        KL = [sum(1 + log(sigma^2) - mu^2 - sigma^2)] / 2
    Args:
        z_mean, z_logvar = torch.chunk(encoder(x), 2, dim=1)
    Returns:
        torch.Tensor: torch.Tensor: scalar
    """
    reconstruction_loss = torch.sum(((pred - x) ** 2))
    kl_divergence = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
    return reconstruction_loss + kl_divergence


class Vae(nn.Module):
    def __init__(self, latent_dim=20, 
                 image_shape=(1, 28, 28),
                w_init_method="xavier", 
                device="cuda"):
        
        super(Vae, self).__init__()
        self.image_shape = image_shape
        self.device = device
        self.features = np.prod(self.image_shape)
        self.latent_dim = latent_dim

        # encoder <==> q(z|x)
        self.encoder = nn.Sequential(
            nn.Linear(self.features, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, self.latent_dim * 2)
        )

        # decoder <==> p(x|z)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, self.features),
            nn.Sigmoid()
        )

        for layer in self.encoder:
            self.initialize_weights(layer, method=w_init_method)
        for layer in self.decoder:
            self.initialize_weights(layer, method=w_init_method)
    
    def initialize_weights(self, layer, method):
        if isinstance(layer, nn.Linear):
            if method == 'xavier':
                init.xavier_normal_(layer.weight)
            elif method == 'he':
                init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            init.constant_(layer.bias, 0.0)

    def trick(self, mean, std):
        epsilon = torch.randn_like(std)
        return mean + std * epsilon

    def forward(self, x):
        x = x.view(-1, self.features)
        z = self.encoder(x)
        mu, logvar = torch.chunk(z, 2, dim=1)
        std = torch.exp(0.5 * logvar)
        reparameterized = self.trick(mu, std)
        pred = self.decoder(reparameterized)
        pred = pred.reshape(-1, *self.image_shape)
        return pred, mu, std

    def sampler(self, num_samples=16):
        sample = Variable(torch.randn(num_samples, self.latent_dim))
        sample = sample.to(self.device)
        pred = self.decoder(sample)
        pred = pred.reshape(-1, *self.image_shape)
        pred = torch.asarray(pred)
        return pred


def train(args, model, train_loader, optimizer, device, indices_images):
    train_loss = 0
    batch_losses = []
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
            train_loss += loss.item()
            pbar.update(x.size(0))
            
            # pbar.set_postfix(loss=torch.mean(loss).item())
            if idx % 1000 == 0:
                sampled = model.sampler(num_samples=train_loader.batch_size)
                merged = torch.cat((pred, sampled), dim=0)
                if not os.path.exists(args.path2results): os.makedirs(args.path2results)
                viz(merged, f"{args.path2results}/rec_and_sampled_{idx}.png")
            if idx in indices_images:
                pred = pred.reshape(-1, *args.img_shape).detach()
                images.append(pred.cpu())
            batch_losses.append(loss.item())

    images = torch.cat(images, dim=0)
    return (train_loss / num_samples), batch_losses, images


def val(args, model, val_loader, device, indices_images):
    
    val_loss = 0
    num_samples = len(val_loader.dataset)
    model.eval()
    images = []
    batch_losses = []

    with torch.no_grad(), tqdm(total=num_samples, desc="Validation", unit="batch") as pbar:
        for idx, (x, _) in enumerate(val_loader):
            x = x.to(device)
            pred, mu, std = model(x)
            loss = elbo_loss(x, pred, mu, std).item()
            val_loss += loss
            pbar.update(x.size(0))
            if idx in indices_images:
                pred = pred.reshape(-1, *args.img_shape).detach()
                images.append(pred.cpu())
            batch_losses.append(loss)
    images = torch.cat(images, dim=0)
    return (val_loss / num_samples), batch_losses, images

    
def train_model(model, 
                train_loader, 
                val_loader, 
                optimizer,
                device,
                epochs=10, 
                early_stop_patience=3,
                n_images=5):
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                     mode='min', 
                                                     factor=0.1, 
                                                     patience=5, 
                                                     verbose=True)
    
    indices_images = np.linspace(0, len(train_loader), n_images+1, dtype=int)
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    early_stop_counter = 0
    for epoch in range(epochs):
        train_loss, batch_train_loss, images_train = train(args,
                                                           model, 
                                                           train_loader, 
                                                           optimizer, 
                                                           device, 
                                                           indices_images)
        
        val_loss, batch_val_loss, images_val  = val(args,
                                                    model, 
                                                    val_loader, 
                                                    device, 
                                                    indices_images)
        # scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            save_model(model,
                       args.model_save_path_best_loss_val,
                       f"best_val_loss_{epoch}_vae.pt")
        elif train_loss < best_train_loss:
            best_train_loss = train_loss
            save_model(model, 
                       args.model_save_path_best_loss_train, 
                       f"best_train_loss_{epoch}_vae.pt")
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

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Vae(latent_dim=args.z_dim,
                image_shape=args.img_shape,
                w_init_method=args.w_init_method,
                device=device)

    print("Device:", device)
    model = model.to(device)

    train_loader, val_loader = get_data_loaders(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    train_model(model, 
                train_loader, 
                val_loader, 
                optimizer,
                device,
                epochs=args.epochs, 
                early_stop_patience=args.early_stop_patience,
                n_images=args.n_images)
    

if __name__ == "__main__":
    args = get_args()
    main(args)