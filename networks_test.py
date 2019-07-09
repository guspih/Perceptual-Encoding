import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from networks import FourLayerCVAE, run_epoch, run_training
import math
import datetime
import os

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
    if data[batch].is_cuda:
        r = r.cuda()
    img1 = torch.index_select(data[batch], 0, r)
    mu, logvar = model.encode(img1)
    z = model.sample(mu,logvar)
    img2 = model.decode(z)
    show(
        [img1.cpu(),img2.cpu()], block=block, 
        heading="Random image from: Epoch {}, Batch {}".format(epoch, batch)
    )

def load_npz_data(data_file, data_size, batch_size,
    split_distribution=[0.8,0.2], gpu=False
):
    '''
    Loads data from an .npz file
    If the data can be turned into a torch tensor, it will
    Anything in key 'imgs' will be loaded as images
    Keys starting with 'parameter' will disregared

    Args:
        data_file (str): Path to file with data
        data_size (int): Amount of data to load
        batch_size (int): How much data will be stored in each batch
        split_distribution ([float]): List of fractions of data split
        gpu (bool): Whether data will be used on the GPU

    Returns: ([{data}])
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
        if (
            isinstance(value[0], np.ndarray) and 
            value[0].dtype.kind in ["f","u","i"]
        ):
            value = [torch.from_numpy(batch) for batch in value]
            if gpu:
                value = [batch.cuda() for batch in value]
        
        data[key] = value

    split_sum = sum(split_distribution)
    split_distribution = [math.floor((data_size/batch_size)*(f/split_sum)) for f in split_distribution]
    j = 0
    while sum(split_distribution) < data_size/batch_size:
        split_distribution[j] = split_distribution[j]+1
        j = j+1
    splits = [{} for _ in range(len(split_distribution))]

    for key, value in data.items():
        if key[:9] == "parameter":
            continue
        for split in splits:
            split[key] = []
        next_up = split_distribution[0]
        j = 0 
        for i, batch in enumerate(value):
            if i == next_up:
                j = j+1
                next_up = next_up+split_distribution[j]
            splits[j][key].append(batch)

    return splits

def train_autoencoder(data, network, epochs, experiment_name="",
    z_dimensions=32, variational=False, gamma=0.001,
    perceptual_loss=False, gpu=False, display=False,
    save_path="autoencoder_checkpoints"
):
    '''
    Trains an image autoencoder with data in dicts with images in key "imgs"

    Args:
        data ([{train},{validation}]): Batched dicts with images in key "imgs"
        network (f()->nn.Module / str): Class or path to a network to train
        epochs (int): Number of epochs to run
        experiment_name (str): Name of experiment
        z_dimensions (int): Number of latent dimensions for encoding
        variational (bool): Whether to train the network as a variational AE
        gamma (float): Weight of the KLD loss in training variational AE
        perceptual_loss (bool): Whether to use perceptual or pixelwise loss
        gpu (bool): Whether to train on the GPU
        display (bool): Whether to display the recreated images
        save_path (str): Path to folder where the trained network will be stored

    Returns (nn.Module, str): The trained autoencoder and file path of the model
    '''
    train_data, validation_data = data
    train_data = train_data["imgs"]
    validation_data = validation_data["imgs"]

    if isinstance(network, str):
        if gpu:
            model = torch.load(network)
        else:
            model = torch.load(network, map_location="cpu")
    else:
        model = network(
            input_size=(64,64),
            z_dimensions=z_dimensions,
            variational=variational,
            gamma=gamma,
            perceptual_loss=perceptual_loss
        )
    if gpu:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters())

    epoch_update = None
    if display:
        epoch_update = lambda _a, _b, _c : show_recreation(
                train_data, model, epoch=0, batch=0, block=False
            ) or True

    if epochs != 0:
        print(
            "Starting autoencoder training. Best checkpoint stored in {}".format(
                save_path
            )
        )   
        model_file = run_training(
            model = model,
            train_data = (train_data, train_data),
            val_data = (validation_data, validation_data),
            loss = model.loss,
            optimizer = optimizer,
            save_path = save_path,
            epochs = epochs,
            epoch_update = epoch_update
        )
    elif isinstance(network, str):
        model_file = network
    else:
        model_file = None

    if display:
        for batch_id in range(len(train_data)):
            show_recreation(
                train_data, model, epoch=epochs, batch=batch_id, block=True
            )
    
    return model, model_file

if __name__ == "__main__":
    '''
    If run directly this will train an autoencoder on the lunarlander dataset

    Parameters:
        DATA_FILE (str): Path to the .npz file with images stored in key "imgs"
        EXPERIMENT_NAME (str): Name of experiment
        EPOCHS (int): Number of epochs to run
        DATA_SIZE (int): Number of images from the data to train on
        BATCH_SIZE (int): Number of images per batch
        VALIDATION_SPLIT (float): Fraction of batches for validation
        Z_DIMENSIONS (int): Number of latent dimensions for encoding
        VARIATIONAL (bool): Whether to train the network as a variational AE
        GAMMA (float): Weight of the KLD loss in training variational AE
        PERCEPTUAL_LOSS (bool): Whether to use with pereptual or pixelwise error
        LOAD_FILE (str / None): Path for loading models, overrides parameters
        GPU (bool): Whether to train on the GPU
        DISPLAY (bool): Whether to display the recreated images
    '''

    DATA_FILE = "LunarLander-v2_105000_Dataset.npz"
    EXPERIMENT_NAME = "LunarLander"
    NETWORK = FourLayerCVAE
    EPOCHS = 100
    DATA_SIZE = 50000
    BATCH_SIZE = 1000
    SPLITS = [0.4, 0.1]
    Z_DIMENSIONS = 32
    VARIATIONAL = False
    GAMMA = 0.001
    PERCEPTUAL_LOSS = False
    GPU = torch.cuda.is_available()
    DISPLAY = False
    SAVE_PATH = "autoencoder_checkpoints"

    DATA = load_npz_data(
        DATA_FILE, DATA_SIZE, BATCH_SIZE, split_distribution=SPLITS, gpu=GPU
    )

    train_autoencoder(
        DATA, NETWORK, EPOCHS, EXPERIMENT_NAME, Z_DIMENSIONS, VARIATIONAL,
        GAMMA, PERCEPTUAL_LOSS, GPU, DISPLAY, SAVE_PATH
    )