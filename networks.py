import random
import torch
import numpy as np
import torchvision.models as models
from torch.nn import functional as F
import torch.nn as nn


def _create_coder(channels, kernel_sizes,
    strides, conv_types, activation_types
):
    '''
    Function that creates en- or decoders based on parameters

    Args:
        channels ([int]): Channel sizes per layer. 1 more than layers
        kernel_sizes ([int]): Kernel sizes per layer
        strides ([int]): Strides per layer
        conv_types ([type]/type): Type of the convoultion module per layer
        activation_types ([type]/type): Type of activation function per layer
    
    Returns: (nn.Sequential) The created coder
    '''
    if not isinstance(conv_types, list):
        conv_types = [conv_types for _ in range(len(kernel_sizes))]

    if not isinstance(activation_types, list):
        activation_types = [activation_types for _ in range(len(kernel_sizes))]

    coder = nn.Sequential()
    for layer in range(len(channels)-1):
        coder.add_module(
            'conv'+ str(layer), 
            conv_types[layer](
                in_channels=channels[layer], 
                out_channels=channels[layer+1],
                kernel_size=kernel_sizes[layer],
                stride=strides[layer]
            )
        )
        coder.add_module(
            'norm'+str(layer),
            nn.BatchNorm2d(channels[layer+1])
        )
        coder.add_module('acti'+str(layer),activation_types[layer]())

    return coder

def run_epoch(network, data, labels, loss, optimizer,
    epoch_name="Epoch", train=True
):
    '''
    Trains a given network for one epoch

    Args:
        network (nn.Module): The network to be trained
        data ([tensor]): List of the batches of data to train on
        labels ([tensor]): List of the batches of labels to target
        loss (f(tensor, tensor)->[tensor]): Loss calculation function
        optimizer (optim.Optimizer): Optimizer for use in training
        epoch_name (str): Name of the epoch (usually a number)
        train (bool): Whether to run this epoch to train or just to evaluate
    
    Returns: (float) The sum of loss of the epoch
    '''
    if train:
        network.train()
    batches = list(zip(data, labels))
    epoch_losses = []
    for batch_id, (batch_data, batch_labels) in enumerate(batches):
        if train:
            optimizer.zero_grad()
        output = network(batch_data)
        losses = loss(output, batch_labels)
        if batch_id == 0:
            epoch_losses = losses
        else:
            epoch_losses = [
                epoch_losses[i] + losses[i] for i in range(len(losses))
            ]
        if train:
            losses[0].backward()
            optimizer.step()
        print(
            "\r{} - [{}/{}] - Losses: {}".format(
                epoch_name, batch_id+1, len(batches),
                ["{0:.5f}".format(l.item()) for l in losses]
            ),end=""
        )
    print()

    return epoch_losses


class CVAE_64x64(nn.Module):
    '''
    A Convolutional Variational Autoencoder for 64x64 images
    
    Args:
        z_dimensions (int): The number of latent dimensions in the encoding
        variational (bool): Whether the model is variational or not
        gamma (float): The weight of the KLD loss
        perceptual_loss: Whether to use pixelwise or AlexNet for recon loss
    '''

    def __init__(self, z_dimensions=32, variational=True, 
        gamma=20.0, perceptual_loss=False
    ):
        super(CVAE_64x64, self).__init__()

        #Attributes
        self.z_dimensions = z_dimensions
        self.variational = variational
        self.gamma = gamma
        self.perceptual_loss = perceptual_loss

        if perceptual_loss:
            self.alexnet = AlexNet(sigmoid_out=True) #TODO: How to normalize losses for fair comparison
        
        encoder_channels = [3,32,64,126,256]

        self.encoder = _create_coder(
            [3,32,64,126,256], [4,4,4,4], [2,2,2,2],
            nn.Conv2d, nn.ReLU
        )
        
        self.mu = nn.Linear(encoder_channels[-1]*4,self.z_dimensions)
        self.logvar = nn.Linear(encoder_channels[-1]*4,self.z_dimensions)

        self.dense = nn.Linear(self.z_dimensions,1024)

        self.decoder = _create_coder(
            [1024,128,64,32,3], [5,5,6,6], [2,2,2,2], 
            nn.ConvTranspose2d,
            [nn.ReLU,nn.ReLU,nn.ReLU,nn.Sigmoid]
        )

        self.relu = nn.ReLU()

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0),-1)
        mu = self.mu(x)
        logvar = self.logvar(x)

        return mu, logvar

    def sample(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
        out = eps.mul(std).add_(mu)
        return out

    def decode(self, z):
        y = self.dense(z)
        y = self.relu(y)
        y = y.view(y.size(0), 1024, 1, 1)
        y = self.decoder(y)
        return y
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        if self.variational:
            z = self.sample(mu, logvar)
        else:
            z = mu
        x_rec = self.decode(z)
        return x_rec, z, mu, logvar

    def loss(self, output, x):
        rec_x, z, mu, logvar = output
        if self.perceptual_loss:
            x = self.alexnet(x)
            rec_x = self.alexnet(rec_x)
        else:
            x = x.reshape(-1, 64 * 64 * 3)
            rec_x = rec_x.view(-1, 64 * 64 * 3)
        REC = F.mse_loss(rec_x, x, reduction='mean')

        if self.variational:
            KLD = -1 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            KLD = torch.tensor([0.0])
        
        return REC + self.gamma*KLD, REC, KLD


class AlexNet(nn.Module):
    '''
    The first layers of Torchvision's pretrained AlexNet

    Args:
        layer (int): The layer from which the features are extracted
        frozen (bool): Whether the network can be trained or not
        sigmoid_out: Whether to add a nn.Sigmoid layer to normalize output 
    '''
    def __init__(self, layer=5, frozen=True, sigmoid_out=True):
        super(AlexNet, self).__init__()
        original_model = models.alexnet(pretrained=True)
        self.features = nn.Sequential(
            *list(original_model.features.children())[:layer]
        )
        if sigmoid_out:
            self.features.add_module('sigmoid',nn.Sigmoid())
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

