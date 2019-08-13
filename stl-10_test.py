import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
from networks import FourLayerCVAE, PrintLogger
from networks import run_epoch, run_training, dense_net
from networks_test import train_autoencoder, dict_to_batches
import math
import datetime
import os

# paths to data
UNLABELED_DATA_PATH = './stl-10/data/stl10_binary/unlabeled_X.bin'
TRAIN_DATA_PATH = './stl-10/data/stl10_binary/train_X.bin'
TRAIN_LABEL_PATH = './stl-10/data/stl10_binary/train_y.bin'
TEST_DATA_PATH = './stl-10/data/stl10_binary/test_X.bin'
TEST_LABEL_PATH = './stl-10/data/stl10_binary/test_y.bin'

def read_stl_labels(path_to_labels):
    '''
    Reads and returns the labels for the STL-10 dataset

    Args:
        path_to_labels (str): Path to the binary file containing labels
    
    Returns ([int]): An array containing the labels
    '''
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        labels = labels-1
        labels = np.eye(10)[labels]
        return labels


def read_stl_images(path_to_data):
    '''
    Reads and returns the images for the STL-10 dataset

    Args:
        path_to_data (str): Path to the binary file containing images

    Returns (np.ndarray()): The images in column-major order
    '''

    with open(path_to_data, 'rb') as f:
        everything = np.fromfile(f, dtype=np.uint8)
        images = np.reshape(everything, (-1, 3, 96, 96))
        images = images/255
        return images

def stl10_experiment(
    encoder, classifier, save_path="stl10_experiments", data_size=90000,
    batch_size=1000, splits=[0.8, 0.2], epochs=20, gpu=False,
    classifier_activations=nn.ReLU
):
    '''
    Trains a classifier on the stl dataset using a CVAE trained on the unlabeled
    data and a dense network that classifies using that CVAE

    Args:
        encoder (str / nn.Module): Encoder or path to encoder used for training
        classifier (str / [int]): Path or neurons per layer of classifier
        save_path (str): Path to folder where to store trained models
        data_size (int): Number of entries to use from the dataset
        batch_size (int): Size of each batch
        splits ([float]): List of two fractions for train and validation split
        epochs (int): Number of training epochs
        gpu (bool): Whether to train on the GPU
        classifier_activations ([f()->nn.Module]): List of activation functions
    '''

    # Load encoder from file or train from scratch
    if isinstance(encoder, str):
        encoder = torch.load(encoder)
    else:
        unlabeled_data = {"imgs" : read_stl_images(UNLABELED_DATA_PATH)}
        unlabeled_data = dict_to_batches(
            unlabeled_data, data_size, batch_size, splits
        )
        encoder, _ = train_autoencoder(
            unlabeled_data, encoder, 100,
            input_size=(96,96), z_dimensions=256, gpu=gpu,
            save_path=save_path
        )
    
    # Load classifier from file or initialize new classifier
    if isinstance(classifier, str):
        classifier = torch.load(encoder)
    else:
        classifier = dense_net(
            encoder.z_dimensions, classifier, classifier_activations
        )
     
    if gpu:
        encoder.cuda()
        classifier.cuda()

    # Load training and test data
    stl_train = {
        "imgs" : read_stl_images(TRAIN_DATA_PATH),
        "labels" : read_stl_labels(TRAIN_LABEL_PATH)
    }
    train_data, validation_data = dict_to_batches(
        stl_train,
        data_size=len(stl_train["imgs"]),
        batch_size=min(len(stl_train["imgs"]),1000),
        split_distribution=splits
    )

    stl_test = {
        "imgs" : read_stl_images(TEST_DATA_PATH),
        "labels" : read_stl_labels(TEST_LABEL_PATH)
    }
    test_data, = dict_to_batches(
        stl_test,
        data_size=len(stl_test["imgs"]),
        batch_size=len(stl_test["imgs"]),
        split_distribution=[1]
    )

    # Generate encodings
    encoder.eval()
    with torch.no_grad():
        print("Generating encodings from images...")
        for split in [train_data, validation_data, test_data]:
            split["encodings"] = []
            for i, img_batch in enumerate(split["imgs"]):
                if gpu:
                    img_batch = img_batch.cuda()
                code_batch = encoder.encode(img_batch)
                if gpu:
                    code_batch = (code_batch[0].cpu(), code_batch[1].cpu())
                    img_batch = img_batch.cpu()
                split["encodings"].append(code_batch[0])

    # Initialize optimizer and loss functions for classifier
    optimizer = torch.optim.Adam(classifier.parameters())
    loss_function = torch.nn.MSELoss()
    loss_functions = lambda output, target : [loss_function(output, target)]

    # Train classifier
    if epochs != 0 or not isinstance(classifier, str):
        classifier, classifier_path = run_training(
            model = classifier, 
            train_data = (train_data["encodings"], train_data["labels"]), 
            val_data = (validation_data["encodings"], validation_data["labels"]), 
            loss = loss_functions,
            optimizer = optimizer,
            save_path = save_path,
            epochs = epochs
        )
    
    # Test classifier
    test_losses = lambda output, target : [
        loss_function(output, target),
        torch.mean(torch.abs(output-target)),
        torch.mean(torch.eq(torch.max(output,1)[1], torch.max(target,1)[1]).float())
    ]

    losses = run_epoch(
        classifier,
        test_data["encodings"],
        test_data["labels"],
        test_losses,
        optimizer,
        "Test",
        False
    )
    print(losses)

if __name__ == "__main__":
    DATA_FILE = UNLABELED_DATA_PATH
    EXPERIMENT_NAME = "STL10"
    NETWORK = FourLayerCVAE
    EPOCHS = 4
    DATA_SIZE = 50000
    BATCH_SIZE = 1000
    SPLITS = [0.4, 0.1]
    Z_DIMENSIONS = 256
    VARIATIONAL = False
    GAMMA = 0.001
    PERCEPTUAL_LOSS = False
    GPU = torch.cuda.is_available()
    DISPLAY = True
    SAVE_PATH = "stl10_experiments"
    INPUT_SIZE = (96,96)
    TRAIN_ONLY_DECODER = True
    CLASSIFIER_LAYERS = [64,16,10]
    CLASSIFIER_ACTIVATIONS = [nn.ReLU, nn.ReLU, nn.Softmax]

    data = {"imgs" : read_stl_images(UNLABELED_DATA_PATH)}
    
    data = dict_to_batches(data, DATA_SIZE, BATCH_SIZE, SPLITS)

    model, path = train_autoencoder(
        data, NETWORK, EPOCHS, EXPERIMENT_NAME, INPUT_SIZE,
        Z_DIMENSIONS, VARIATIONAL, GAMMA, PERCEPTUAL_LOSS,
        GPU, DISPLAY, SAVE_PATH, TRAIN_ONLY_DECODER
    )

    stl10_experiment(
        path,
        CLASSIFIER_LAYERS,
        gpu=GPU,
        batch_size=BATCH_SIZE,
        data_size=DATA_SIZE,
        epochs=1000,
        classifier_activations=CLASSIFIER_ACTIVATIONS
    )

