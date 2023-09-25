import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import CelebA, MNIST
from tqdm import tqdm
import torchinfo

import sys; sys.path.append("../utils")
from utils import create_gif_from_folder, viz, plot_losses, save_model, get_args, get_models, get_data_loaders, log

import wandb
wandb.init(project="vae")


def elbo_loss(x, pred, z_mean, z_logvar, beta):
    """
    log(p(x)) = KL(q(z|x) || p(z)) + elbo 
    mse - kl = elbo
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
    # TODO save reconstruction_loss and kl_divergence separately to plot them later
    wandb.log({'elbo':reconstruction_loss + kl_divergence,
               'reconstruction_loss': reconstruction_loss, 
               'kl_divergence': beta * kl_divergence})
    return reconstruction_loss +  beta * kl_divergence


def train(args, model, train_loader, optimizer, device, indices_images, beta):
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
            loss = elbo_loss(x, pred, mu, std, beta)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.update(x.size(0))
            log(args, model, train_loader, x, pred, idx, images, indices_images)
            batch_losses.append(loss.item())

    images = torch.cat(images, dim=0)
    return (train_loss / num_samples), batch_losses, images


def val(args, model, val_loader, device, indices_images, beta):
    
    val_loss = 0
    num_samples = len(val_loader.dataset)
    model.eval()
    images = []
    batch_losses = []

    with torch.no_grad(), tqdm(total=num_samples, desc="Validation", unit="batch") as pbar:
        for idx, (x, _) in enumerate(val_loader):
            x = x.to(device)
            pred, mu, std = model(x)
            loss = elbo_loss(x, pred, mu, std, beta).item()
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
    beta = 1
    for epoch in range(epochs):
        if epoch > 0:
            beta = 15
        print(f"\nBeta {beta} epoch {epoch} \n")
        train_loss, batch_train_loss, images_train = train(args,
                                                           model, 
                                                           train_loader, 
                                                           optimizer, 
                                                           device, 
                                                           indices_images,
                                                           beta)
        
        val_loss, batch_val_loss, images_val  = val(args,
                                                    model, 
                                                    val_loader, 
                                                    device, 
                                                    indices_images, 
                                                    beta)
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
                   'best_val_loss': best_val_loss},
                    step=epoch+1)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_models(args, device)

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

    summary = torchinfo.summary(model, input_size=tuple(args.img_shape))
    with open(f"{args.path2results}/model.txt", "w") as f:
        f.write(str(summary))

if __name__ == "__main__":
    args = get_args()
    main(args)
