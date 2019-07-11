import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from networks import FourLayerCVAE, run_epoch, run_training
from networks_test import train_autoencoder
import math
import datetime
import os

# paths to data
UNLABELED_DATA_PATH = './stl-10/data/stl10_binary/unlabeled_X.bin'
TRAIN_DATA_PATH = './stl-10/data/stl10_binary/train_X.bin'
TRAIN_LABEL_PATH = './stl-10/data/stl10_binary/train_y.bin'
TEST_DATA_PATH = './stl-10/data/stl10_binary/test_X.bin'
TEST_LABEL_PATH = './stl-10/data/stl10_binary/test_y.bin'

def read_labels(path_to_labels):
    '''
    Reads and returns the labels for the STL-10 dataset

    Args:
        path_to_labels (str): Path to the binary file containing labels
    
    Returns ([int]): An array containing the labels
    '''
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels


def read_images(path_to_data):
    '''
    Reads and returns the images for the STL-10 dataset

    Args:
        path_to_data (str): Path to the binary file containing images

    Returns (np.ndarray()): The images in column-major order
    '''

    with open(path_to_data, 'rb') as f:
        everything = np.fromfile(f, dtype=np.uint8)
        images = np.reshape(everything, (-1, 3, 96, 96))
        return images

def imgs2batches(npimgs, data_size, batch_size, split_distribution, gpu=False):
    '''
    Takes an nparray of images and creates a list of dicts with images in "imgs"

    Args:
        npimgs (nparray): Four dimensional nparray of images
        data_size (int): How many entries will be used
        batch_size (int): Size of batches (data_size must be divisible by this)
        split_distribution ([float]): List of fractions of data split
        gpu (bool): Whether data will be used on the GPU

    Returns ([{data}]): A list of dicts where dict["imgs"] is a batch
    '''

    assert data_size % batch_size == 0, \
        "data_size must be divisble by batch_size"
    
    npimgs = npimgs[:data_size]

    npimgs = np.split(npimgs, data_size/batch_size)

    split_sum = sum(split_distribution)
    split_distribution = [math.floor((data_size/batch_size)*(f/split_sum)) for f in split_distribution]
    j = 0
    while sum(split_distribution) < data_size/batch_size:
        split_distribution[j] = split_distribution[j]+1
        j = j+1
    splits = [{} for _ in range(len(split_distribution))]

    for split in splits:
        split["imgs"] = []
    next_up = split_distribution[0]
    j = 0 
    for i, batch in enumerate(npimgs):
        if i == next_up:
            j = j+1
            next_up = next_up+split_distribution[j]
        batch = batch/255
        batch = torch.from_numpy(batch)
        batch = batch.float()
        if gpu:
            batch = batch.cuda()
        splits[j]["imgs"].append(batch)

    return splits

if __name__ == "__main__":
    DATA_FILE = UNLABELED_DATA_PATH
    EXPERIMENT_NAME = "STL10"
    NETWORK = "stl10_experiments/FourLayerCVAE_2019-07-09_16h03.pt"
    EPOCHS = 100
    DATA_SIZE = 50000
    BATCH_SIZE = 1000
    SPLITS = [0.4, 0.1]
    Z_DIMENSIONS = 64
    VARIATIONAL = False
    GAMMA = 0.001
    PERCEPTUAL_LOSS = False
    GPU = torch.cuda.is_available()
    DISPLAY = True
    SAVE_PATH = "stl10_experiments"
    INPUT_SIZE = (96,96)
    TRAIN_ONLY_DECODER = True

    imgs = read_images(DATA_FILE)

    data = imgs2batches(imgs, DATA_SIZE, BATCH_SIZE, SPLITS, GPU)

    train_autoencoder(
        data, NETWORK, EPOCHS, EXPERIMENT_NAME, INPUT_SIZE,
        Z_DIMENSIONS, VARIATIONAL, GAMMA, PERCEPTUAL_LOSS,
        GPU, DISPLAY, SAVE_PATH, TRAIN_ONLY_DECODER
    )