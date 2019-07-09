import random
import torch
import numpy as np
import torchvision.models as models
from torch.nn import functional as F
import torch.nn as nn
import datetime
import time
import sys


def _create_coder(channels, kernel_sizes, strides,
    conv_types, activation_types, paddings = (0,0)
):
    '''
    Function that creates en- or decoders based on parameters

    Args:
        channels ([int]): Channel sizes per layer. 1 more than layers
        kernel_sizes ([int]): Kernel sizes per layer
        strides ([int]): Strides per layer
        conv_types ([f()->type]): Type of the convoultion module per layer
        activation_types ([f()->type]): Type of activation function per layer
        paddings ([(int, int)]): The padding per layer
    
    Returns: (nn.Sequential) The created coder
    '''
    if not isinstance(conv_types, list):
        conv_types = [conv_types for _ in range(len(kernel_sizes))]

    if not isinstance(activation_types, list):
        activation_types = [activation_types for _ in range(len(kernel_sizes))]

    if not isinstance(paddings, list):
        paddings = [paddings for _ in range(len(kernel_sizes))]


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
        loss (f(output, target)->[tensor]): Loss calculation function
        optimizer (optim.Optimizer): Optimizer for use in training
        epoch_name (str): Name of the epoch (usually a number)
        train (bool): Whether to run this epoch to train or just to evaluate
    
    Returns: (float) The sum of loss of the epoch
    '''
    start_time = time.time()

    if train:
        network.train()
    else:
        network.eval()
    batches = list(zip(data, labels))
    epoch_losses = []
    for batch_id, (batch_data, batch_labels) in enumerate(batches):
        optimizer.zero_grad()
        output = network(batch_data)
        losses = loss(output, batch_labels)
        if batch_id == 0:
            epoch_losses = [
                loss.item() for loss in losses
            ]
        else:
            epoch_losses = [
                epoch_losses[i] + losses[i].item() for i in range(len(losses))
            ]
        losses[0].backward()
        if train:
            optimizer.step()
        print(
            "\r{} - [{}/{}] - Losses: {}, Time passed: {}s".format(
                epoch_name, batch_id+1, len(batches),
                ", ".join(
                    ["{0:.5f}".format(l/(batch_id+1)) for l in epoch_losses]
                ),
                "{0:.1f}".format(time.time()-start_time)
            ),end=""
        )
    print()

    return epoch_losses

def run_training(model, train_data, val_data, loss,
    optimizer, save_path, epochs, epoch_update=None
):
    '''
    Args:
        model (nn.Module): The network to be trained
        train_data ([tensor],[tensor]): Batches of data and labels to train on
        val_data ([tensor],[tensor]): Batches of data and labels to validate on
        loss (f(output, target)->[tensor]): Loss calculation function
        optimizer (optim.Optimizer): Optimizer for use in training
        save_path (str): Path to folder where the model will be stored
        epochs (int): Number of epochs to train for
        epoch_update (f(epoch, train_loss, val_loss) -> bool): Function to run
            at the end of a epoch. Returns whether to early stop

    Returns (str): The filepath to the new model
    '''

    save_file = (
        model. __class__.__name__ + 
        datetime.datetime.now().strftime("_%Y-%m-%d_%Hh%M.pt")
    )
    if save_path != "":
        save_file = save_path + "/" + save_file

    torch.save(model, save_file)
    best_validation_loss = float("inf")
    for epoch in range(1,epochs+1):
        training_losses = run_epoch(
            model, train_data[0], train_data[1], loss, optimizer,
            "Train {}".format(epoch), train=True
        )

        validation_losses = run_epoch(
            model, val_data[0], val_data[1], loss, optimizer,
            "Validation {}".format(epoch), train=False
        )
        
        if validation_losses[0] < best_validation_loss:
            torch.save(model, save_file)
            best_validation_loss = validation_losses[0]
        
        if not epoch_update is None:
            early_stop = epoch_update(epoch, training_losses, validation_losses)
            if early_stop:
                break

    return save_file

class PrintLogger():
    '''
    Logger used for logging printouts

    Args:
        logs ([str]): Files to store logs in
    '''
    
    def __init__(self, logs):
        self.stream = sys.stdout
        self.targets = [open(log, "a+") for log in logs]
        self.targets.append(self.stream)
        sys.stdout = self

    def write(self, text):
        [t.write(text) for t in self.targets]

    def flush(self):
        [t.flush() for t in self.targets]

    def __del__(self):
        sys.stdout = self.stream


class FourLayerCVAE(nn.Module):
    '''
    A Convolutional Variational Autoencoder for images
    
    Args:
        input_size (int,int): The height and width of the input image
            acceptable sizes are 64+16*n
        z_dimensions (int): The number of latent dimensions in the encoding
        variational (bool): Whether the model is variational or not
        gamma (float): The weight of the KLD loss
        perceptual_loss: Whether to use pixelwise or AlexNet for recon loss
    '''

    def __init__(self, input_size=(64,64), z_dimensions=32,
        variational=True, gamma=20.0, perceptual_loss=False
    ):
        super(FourLayerCVAE, self).__init__()

        #Parameter check
        if (input_size[0] - 64) % 16 != 0 or (input_size[1] - 64) % 16 != 0:
            raise ValueError(
                "Input_size is {}, but must be 64+16*N".format(input_size)
            )

        #Attributes
        self.input_size = input_size
        self.z_dimensions = z_dimensions
        self.variational = variational
        self.gamma = gamma
        self.perceptual_loss = perceptual_loss

        if perceptual_loss:
            self.alexnet = AlexNet(sigmoid_out=True)
        
        encoder_channels = [3,32,64,126,256]

        self.encoder = _create_coder(
            [3,32,64,126,256], [4,4,4,4], [2,2,2,2],
            nn.Conv2d, nn.ReLU
        )
        
        f = lambda x: np.floor((x - (2,2))/2)
        conv_sizes = f(f(f(f(np.array(input_size)))))
        conv_flat_size = int(encoder_channels[-1]*conv_sizes[0]*conv_sizes[1])
        self.mu = nn.Linear(conv_flat_size, self.z_dimensions)
        self.logvar = nn.Linear(conv_flat_size, self.z_dimensions)

        self.dense = nn.Linear(self.z_dimensions, conv_flat_size)

        self.decoder = _create_coder(
            [1024,128,64,32,3], [5,5,6,6], [2,2,2,2],
            nn.ConvTranspose2d,
            [nn.ReLU,nn.ReLU,nn.ReLU,nn.Sigmoid]
        )

        self.relu = nn.ReLU()

    def __str__(self):
        string = super(FourLayerCVAE, self).__str__()[:-1]
        string = string + "  (variational): {}\n  (gamma): {}\n)".format(
                self.variational,self.gamma
            )
        return string

    def __repr__(self):
        string = super(FourLayerCVAE, self).__repr__()[:-1]
        string = string + "  (variational): {}\n  (gamma): {}\n)".format(
                self.variational,self.gamma
            )
        return string

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
        y = y.view(
            y.size(0), 1024,
            int((self.input_size[0]-64)/16)+1,
            int((self.input_size[0]-64)/16)+1
        )
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
            if REC.is_cuda:
                KLD = KLD.cuda()

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

def dense_net(input_size, layers, activation_functions):
    '''
    Creates a simple dense network

    Args:
        input_size (int): Input size to the network
        layers ([int]): Layer sizes
        activation_functions ([f()->nn.Module]): class of activation functions
    
    Returns: (nn.Sequential)
    '''

    network = nn.Sequential()
    layers = [input_size] + layers
    for layer_id in range(len(layers)-1):
        network.add_module(
            'linear{}'.format(layer_id),
            nn.Linear(layers[layer_id], layers[layer_id+1])
        )
        if not activation_functions[layer_id] is None:
            network.add_module(
                'activation{}'.format(layer_id),
                activation_functions[layer_id]()
            )
    return network