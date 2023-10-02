import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable


class Vae(nn.Module):
    def __init__(self, 
                    enc_backbone,
                    dec_backbone, 
                    latent_dim=20, 
                    image_shape=(1, 28, 28), 
                    w_init_method="xavier", 
                    device="cuda"):
        
        super(Vae, self).__init__()
        self.image_shape = image_shape
        self.device = device
        self.features = np.prod(self.image_shape)
        self.latent_dim = latent_dim

        self.encoder = enc_backbone
        self.decoder = dec_backbone

        for layer in self.encoder:
            self.initialize_weights(layer, method=w_init_method)
        for layer in self.decoder:
            self.initialize_weights(layer, method=w_init_method)
        
    
    def initialize_weights(self, layer, method):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Linear):
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
        return pred, sample
    
        
class Lvae(Vae):
    def __init__(self, 
                 latent_dim=20, 
                 image_shape=(1, 28, 28), 
                 w_init_method="xavier", 
                 device="cuda"):
        
        self.features = np.prod(image_shape)
        enc_backbone = nn.Sequential(
            nn.Linear(self.features, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, latent_dim * 2)
        )

        dec_backbone = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, self.features),
            nn.Sigmoid()
        )

        super().__init__(enc_backbone, 
                         dec_backbone,  
                         latent_dim=latent_dim, 
                         image_shape=image_shape, 
                         w_init_method=w_init_method, 
                         device=device)

class PCvae(nn.Module):
    def __init__(self, latent_dim=20, 
                 image_shape=(1, 28, 28),
                 w_init_method="xavier", 
                 device="cuda"):
        
        super(PCvae, self).__init__()
        self.image_shape = image_shape
        self.device = device
        self.features = np.prod(self.image_shape)
        self.latent_dim = latent_dim

        # encoder <==> q(z|x)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3), 
            nn.LeakyReLU(0.4),
            nn.MaxPool2d(2, 2),  
            nn.Conv2d(128, 64, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 32, kernel_size=3, stride=2),
            nn.LeakyReLU()
        )

        # decoder <==> p(x|z)
        self.decoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.6),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.3),
            nn.Linear(512, self.features),
            nn.Sigmoid()
        )

        for layer in self.encoder:
            self.initialize_weights(layer, method=w_init_method)
        for layer in self.decoder:
            self.initialize_weights(layer, method=w_init_method)
    
    def initialize_weights(self, layer, method):
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            if method == 'xavier':
                init.xavier_normal_(layer.weight)
            elif method == 'he':
                init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            init.constant_(layer.bias, 0.0)

    def trick(self, mean, std):
        epsilon = torch.randn_like(std)
        return mean + std * epsilon

    def forward(self, x):
        z = self.encoder(x) # 128, 128, 2, 2
        #print(f"z shape {z.shape}")
        z = z.reshape(z.shape[0], -1)
        #print(f"z shape {z.shape}")
        mu, logvar = torch.chunk(z, 2, dim=1) # 32, 32
        std = torch.exp(0.5 * logvar)
        #print(f"mu shape {mu.shape}")
        reparameterized = self.trick(mu, std) # 128, 32
        pred = self.decoder(reparameterized) # 128, 46875
        pred = pred.reshape(-1, *self.image_shape) # 128, 3, 125, 125
        return pred, mu, std

    def sampler(self, num_samples=16):
        sample = Variable(torch.randn(num_samples, 784))
        sample = sample.to(self.device)
        pred = self.decoder(sample)
        pred = pred.reshape(-1, *self.image_shape)
        pred = torch.asarray(pred)
        return pred
    

class FCvae(nn.Module):
    def __init__(self, latent_dim=20, 
                 image_shape=(1, 28, 28),
                 w_init_method="xavier", 
                 device="cuda"):
        
        super(FCvae, self).__init__()
        self.image_shape = image_shape
        self.device = device
        self.features = np.prod(self.image_shape)
        self.latent_dim = latent_dim

        # encoder <==> q(z|x)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3), 
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),  
            nn.Conv2d(64, 32, kernel_size=3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 2, kernel_size=3),
            nn.LeakyReLU(),
            nn.AvgPool2d(4, 4),
            nn.Sigmoid()
        )

        # decoder <==> p(x|z)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 32, kernel_size=4, stride=3),  
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 64, kernel_size=5, stride=2), 
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=5, stride=3),  
            nn.Sigmoid(),
        )

        for layer in self.encoder:
            self.initialize_weights(layer, method=w_init_method)
        for layer in self.decoder:
            self.initialize_weights(layer, method=w_init_method)
    
    def initialize_weights(self, layer, method):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
            if method == 'xavier':
                init.xavier_normal_(layer.weight)
            elif method == 'he':
                init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            init.constant_(layer.bias, 0.0)

    def trick(self, mean, std):
        epsilon = torch.randn_like(std)
        return mean + std * epsilon

    def forward(self, x):
        # x 32, 3, 125, 125
        z = self.encoder(x) # 32, 2, 6, 6
        mu, logvar = torch.chunk(z, 2, dim=1) # mu 32, 1, 6, 6, logvar 32, 1, 6, 6
        std = torch.exp(0.5 * logvar)
        reparameterized = self.trick(mu, std) # 1, 1, 6, 6
        pred = self.decoder(reparameterized)  # 32, 3, 125, 125
        return pred, mu, std

    def sampler(self, num_samples=16):
        sample = Variable(torch.randn(num_samples, 36))
        sample = sample.reshape(num_samples, 1, 6, 6)
        sample = sample.to(self.device)
        pred = self.decoder(sample)
        pred = torch.asarray(pred)
        return pred