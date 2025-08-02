"""
CAAE (Conditional Adversarial Autoencoder) implementation for face aging.

Based on the paper: "Face Aging with Conditional Adversarial Autoencoders"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Encoder network for CAAE model."""
    
    def __init__(self, z_dim=50, c_dim=6):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 4, 2, 1)  # 128x128 -> 64x64
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)  # 64x64 -> 32x32
        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)  # 32x32 -> 16x16
        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1)  # 16x16 -> 8x8
        self.conv5 = nn.Conv2d(256, 512, 4, 2, 1)  # 8x8 -> 4x4
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        
        # Fully connected layers
        self.fc_z = nn.Linear(512 * 4 * 4, z_dim)
        self.fc_c = nn.Linear(512 * 4 * 4, c_dim)
        
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.2)
        
        x = x.view(x.size(0), -1)
        z = self.fc_z(x)
        c = self.fc_c(x)
        
        return z, c


class Decoder(nn.Module):
    """Decoder network for CAAE model."""
    
    def __init__(self, z_dim=50, c_dim=6):
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        
        # Fully connected layer
        self.fc = nn.Linear(z_dim + c_dim, 512 * 4 * 4)
        
        # Transposed convolutional layers
        self.deconv1 = nn.ConvTranspose2d(512, 256, 4, 2, 1)  # 4x4 -> 8x8
        self.deconv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)  # 8x8 -> 16x16
        self.deconv3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)   # 16x16 -> 32x32
        self.deconv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)    # 32x32 -> 64x64
        self.deconv5 = nn.ConvTranspose2d(32, 3, 4, 2, 1)     # 64x64 -> 128x128
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)
        
    def forward(self, z, c):
        x = torch.cat([z, c], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), 512, 4, 4)
        
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = F.relu(self.bn4(self.deconv4(x)))
        x = torch.tanh(self.deconv5(x))
        
        return x


class CAAE(nn.Module):
    """Complete CAAE model for face aging."""
    
    def __init__(self, z_dim=50, c_dim=6):
        super(CAAE, self).__init__()
        self.encoder = Encoder(z_dim, c_dim)
        self.decoder = Decoder(z_dim, c_dim)
        
    def forward(self, x, target_age=None):
        z, c = self.encoder(x)
        
        if target_age is not None:
            # Use target age for conditioning
            reconstructed = self.decoder(z, target_age)
        else:
            # Use original age conditioning
            reconstructed = self.decoder(z, c)
            
        return reconstructed, z, c
