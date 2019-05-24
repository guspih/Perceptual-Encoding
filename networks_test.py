import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from networks import CVAE_64x64, run_epoch
import math
import datetime

def show(imgs, block=False, heading="Figure"):
    '''
    Paints a column of torch images

    Args:
        imgs ([3darray]): Array of images in shape (channels, width, height)
        block (bool): Whether the image should interupt program flow
        heading (str)): The heading to put on the image
    '''
    plt.close()
    fig=plt.figure(figsize=(8, 8))
    fig.canvas.set_window_title(heading)
    for i, img in enumerate(imgs):
        img = img[0].detach().permute(1,2,0)
        fig.add_subplot(len(imgs), 1, i+1)
        plt.imshow(img)
    plt.show(block=block)
    plt.pause(5)

def show_recreation(data, model, epoch=0, batch=0, block=False):
    '''
    Shows a random image and the encoders attempted recreation

    Args:
        data ([tensor]): List of image batches
        model (nn.Module): (V)AE model to be run
        epoch (int): Epoch to be printed on the heading
        batch (int): Batch of data to sample the image from
        block (bool): Whether to stop execution until user closes image
    '''
    r = torch.tensor([random.choice(range(data[batch].size(0)))])
    if data.is_cuda:
        r.cuda()
    img1 = torch.index_select(data[batch], 0, r)
    mu, logvar = model.encode(img1)
    z = model.sample(mu,logvar)
    img2 = model.decode(z)
    show(
        [img1,img2], block=block, 
        heading="Random image from: Epoch {}, Batch {}".format(epoch, batch)
    )

def load_npz_data(data_file, data_size, batch_size,
    test_split=0.2, gpu=False
):
    '''
    Loads data from an .npz file
    If the data can be turned into a torch tensor, it will
    Anything in key 'imgs' will be loaded as images
    Keys starting with 'parameter' will disregared

    Args:
        data_file (str): Path to file with data
        data_size (int): Amount of images to load
        batch_size (int): How much data will be stored in each batch
        test_split (float): What portion of batches to use for testing
        gpu (bool): Whether data will be used on the GPU

    Returns: ([tensor],[tensor])
    '''
    assert data_size % batch_size == 0, \
        "data_size must be divisble by batch_size"
    print("Loading data from {}...".format(data_file))
    data = dict(np.load(data_file))
    
    for key, value in data.items():
        if key == "imgs":
            value = value[:data_size]
            value = np.array(value, dtype=np.float32)
            value = np.transpose(value, (0,3,1,2))
            value = np.split(value, value.shape[0]/batch_size)
        elif key[:9] == "parameter":
            continue
        else:
            value = value[:data_size]
            value = np.split(value, value.shape[0]/batch_size)
        if (isinstance(value[0], np.ndarray) and 
                value[0].dtype.kind in ["f","u","i"]):
            value = [torch.from_numpy(batch) for batch in value]
            if gpu:
                value = [batch.cuda() for batch in value]
        
        data[key] = value

    train_data = {}
    test_data = {}
    for key, value in data.items():
        if key[:9] == "parameter":
            continue
        train_data[key] = []
        test_data[key] = []
        for batch in value:
            if len(train_data[key])/len(value) < 1-test_split:
                train_data[key].append(batch)
            else:
                test_data[key].append(batch)

    return train_data, test_data

def train_npz_autoencoder(data_file, network, epochs, data_size, batch_size,
    validation_split, z_dimensions=32, variational=False, gamma=0.001,
    perceptual_loss=False, load_file=None, gpu=False, display=False
):
    '''
    Trains an autoencoder with data from a specified .npz file

    Args:
        data_file (str): Path to the .npz file with images stored with key "imgs"
        network (f()->nn.Module): Class of network to train
        epochs (int): Number of epochs to run
        data_size (int): Number of images from the data to train on
        batch_size (int): Number of images per batch
        validation_split (float): Fraction of batches for validation
        z_dimensions (int): Number of latent dimensions for encoding
        variational (bool): Whether to train the network as a variational AE
        gamma (float): Weight of the KLD loss in training variational AE
        perceptual_loss (bool): Whether to train with pereptual or pixelwise error
        load_file (str / None): Path for loading models, overrides parameters
        gpu (bool): Whether to train on the GPU
        display (bool): Whether to display the recreated images

        Returns: (nn.Module)
    '''
    data, validation_data = load_npz_data(
        data_file, data_size, batch_size, validation_split, gpu=gpu
    )
    data = data["imgs"]
    validation_data = validation_data["imgs"]

    if not load_file is None:
        model = torch.load(load_file)
    else:
        model = network(
            z_dimensions=z_dimensions,
            variational=variational,
            gamma=gamma,
            perceptual_loss=perceptual_loss
        )
    if gpu:
        model = model.cuda()

    save_file = "AE_{}_{}_{}_{}.pt".format(
        data_file, z_dimensions, gamma,
        datetime.datetime.now().strftime("%Y-%m-%d_%Hh%M")
    )
    if variational:
        save_file = "V"+save_file
    if perceptual_loss:
        save_file = "Perceptual_"+save_file

    optimizer = torch.optim.Adam(model.parameters())

    best_validation_loss = float("inf")

    for epoch in range(epochs):
        np.random.shuffle(data)
        training_losses = run_epoch(
            model, data, data, model.loss, optimizer,
            "Epoch {}".format(epoch), train=True
        )

        validation_losses = run_epoch(
            model, validation_data, validation_data, model.loss, optimizer,
            "Validation {}".format(epoch), train=False
        )

        print(
            "EPOCH: {}, TRAINING LOSS: {} VALIDATION LOSS: {}".format(
                epoch, training_losses, validation_losses
            )
        )
        
        if validation_losses[0] < best_validation_loss:
            torch.save(model.cpu(), save_file)
            best_validation_loss = validation_losses[0]
        
        if gpu:
            model.cuda()
        if display:
            show_recreation(data, model, epoch=epoch, batch=0, block=False)
    if display:
        for batch_id in range(len(data)):
            show_recreation(data, model, epoch=epochs, batch=batch_id, block=True)
    
    return model

if __name__ == "__main__":
    '''
    If run directly this will train an autoencoder on the lunarlander dataset

    Parameters:
            DATA_FILE (str): Path to the .npz file with images stored with key "imgs"
            EPOCHS (int): Number of epochs to run
            DATA_SIZE (int): Number of images from the data to train on
            BATCH_SIZE (int): Number of images per batch
            VALIDATION_SPLIT (float): Fraction of batches for validation
            Z_DIMENSIONS (int): Number of latent dimensions for encoding
            VARIATIONAL (bool): Whether to train the network as a variational AE
            GAMMA (float): Weight of the KLD loss in training variational AE
            PERCEPTUAL_LOSS (bool): Whether to train with pereptual or pixelwise error
            LOAD_FILE (str / None): Path for loading models, overrides parameters
            GPU (bool): Whether to train on the GPU
            DISPLAY (bool): Whether to display the recreated images
    '''

    DATA_FILE = "LunarLander-v2_105000_Dataset.npz"
    EPOCHS = 50
    DATA_SIZE = 50000
    BATCH_SIZE = 1000
    VALIDATION_SPLIT = 0.2
    Z_DIMENSIONS = 32
    VARIATIONAL = False
    GAMMA = 0.001
    PERCEPTUAL_LOSS = False
    LOAD_FILE = None 
    GPU = False
    DISPLAY = True

train_npz_autoencoder(
    DATA_FILE, CVAE_64x64, EPOCHS, DATA_SIZE,
    BATCH_SIZE, VALIDATION_SPLIT, Z_DIMENSIONS, VARIATIONAL,
    GAMMA, PERCEPTUAL_LOSS, LOAD_FILE, GPU, DISPLAY
)