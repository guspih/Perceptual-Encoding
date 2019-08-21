import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import math
import datetime
import csv
import os
import sys

from networks import FourLayerCVAE, PrintLogger
from networks import dense_net, run_epoch, run_training
from networks_test import load_npz_data, train_autoencoder

# Experiment recording
EXPERIMENTS_PATH = "lunarlander_experiments"
EXPERIMENTS_FILE = "experiments_data.csv"
try:
    os.makedirs(EXPERIMENTS_PATH)
except FileExistsError:
    pass

# Paths to data
ENCODER_DATA_PATH = "LunarLander-v2_105000_Dataset.npz"
REGRESSOR_DATA_PATH = "LunarLander-v2_92623_Cleaned.npz"


def run_lunarlander_experiment(
    encoder, regressor, save_path="lunarlander_experiments", data_size=90000,
    batch_size=1000,epochs=20, gpu=False, regressor_activations=[None]
):
    '''
    Trains a regressor on the LunarLander dataset using a CVAE trained on the
    data and a dense network that predicts lander position from the encodings

    Args:
        encoder (str / nn.Module): Encoder or path to encoder used for training
        regressor (str / [int]): Neurons per layer or path to regressor
        save_path (str): Path to folder where to store trained models
        data_size (int): Number of entries to use from the dataset
        batch_size (int): Size of each batch
        epochs (int): Number of training epochs
        gpu (bool): Whether to train on the GPU
        regressor_activations ([f()->nn.Module]): List of activation functions
    '''
    experiment_data = locals()
    experiment_time = datetime.datetime.now().strftime("%Y-%m-%d_%Hh%M")
    log_file = "log_"+experiment_time+".log"
    if save_path != "":
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        log_file = save_path+"/"+log_file
    PrintLogger([log_file])
    experiment_data["log_file"] = log_file

    print("Starting LunarLander-v2 regression experiment...")

    if isinstance(encoder, str):
        encoder_file = encoder
        encoder = torch.load(encoder_file)
    else:
        encoder_data = load_npz_data(
            ENCODER_DATA_PATH, data_size, batch_size, [0.8, 0.2]
        )
        encoder, encoder_file = train_autoencoder(
            encoder_data, encoder, 100, input_size=(64,64), z_dimensions=32, 
            gpu=gpu, save_path="LunarLander-Pretraining"
        )

    if isinstance(regressor, str):
        regressor_file = regressor
        regressor = torch.load(regressor)
    else:
        regressor = dense_net(
            encoder.z_dimensions,
            regressor,
            regressor_activations
        )

    if gpu:
        encoder.cuda()
        regressor.cuda()

    train_data, val_data, test_data = load_npz_data(
        REGRESSOR_DATA_PATH, data_size, batch_size, [0.6, 0.2, 0.2]
    )

    encoder.eval()
    with torch.no_grad():
        print("Generating encodings from images...")
        for split in [train_data, val_data, test_data]:
            split["encodings"] = []
            for i, img_batch in enumerate(split["imgs"]):
                if gpu:
                    img_batch = img_batch.cuda()
                code_batch = encoder.encode(img_batch)
                if gpu:
                    code_batch = (code_batch[0].cpu(), code_batch[1].cpu())
                    img_batch = img_batch.cpu()
                split["encodings"].append(code_batch[0])
            new_observations = []
            for i, obs_batch in enumerate(split["observations"]):
                new_observations.append(obs_batch.narrow(1,0,2))
            split["observations"] = new_observations

    optimizer = torch.optim.Adam(regressor.parameters())
    loss_function = torch.nn.MSELoss()
    loss_functions = lambda output, target : [loss_function(output, target)]

    if epochs != 0:
        regressor, regressor_file, val_loss = run_training(
            model = regressor,
            train_data = (train_data["encodings"], train_data["observations"]),
            val_data = (val_data["encodings"], val_data["observations"]),
            loss = loss_functions,
            optimizer = optimizer,
            save_path = save_path,
            epochs = epochs,
            epoch_update = None
        )

    experiment_data["regressor_file"] = regressor_file
    experiment_data["encoder"] = encoder
    experiment_data["regressor"] = regressor

    test_losses = lambda output, target : [
            loss_function(output, target),
            torch.mean(torch.abs(output-target))
        ]
        
    losses = run_epoch(
        regressor,
        test_data["encodings"],
        test_data["observations"],
        test_losses,
        optimizer,
        "Test",
        False
    )

    experiment_data["test_loss"] = losses[0]

    test_accuracy = losses[1]/len(test_data["encodings"])
    print(
        "Test - Loss {:.5f} - Mean Error {:.5f}".format(
            losses[0]/len(test_data["encodings"]), test_accuracy
        )
    )

    experiment_data["test_loss"] = losses[0]
    experiment_data["test_mean_error"] = test_accuracy
    experiment_data["val_loss"] = val_loss

    with open(EXPERIMENTS_PATH+"/"+EXPERIMENTS_FILE, "a+") as f:
        f.write("Experiment at {}\n".format(experiment_time).replace("_", " "))
        f.write(str(experiment_data))
        f.write("\n\n")

    return experiment_data

if __name__ == "__main__":
    '''
    If run directly will train a VAE and
    a regressor from the encodings of that VAE

    Args:
        encoder (str / nn.Module): Encoder or path to encoder used for training
        regressor (str / [int]): Neurons per layer or path to regressor
        save_path (str): Path to folder where to store trained models
        data_size (int): Number of entries to use from the dataset
        batch_size (int): Size of each batch
        epochs (int): Number of training epochs
        gpu (bool): Whether to train on the GPU
        regressor_activations ([f()->nn.Module]): List of activation functions
    '''
    
    ENCODER = "autoencoder_checkpoints/CVAE_64x64_2019-06-28_13h58.pt"
    REGRESSOR = [2]
    SAVE_PATH = "lunarlander_experiments"
    DATA_SIZE = 90000
    BATCH_SIZE = 1000
    EPOCHS = 50
    GPU = torch.cuda.is_available()
    REGRESSOR_ACTIVATIONS = [None]

    run_lunarlander_experiment(
        encoder=ENCODER,
        regressor=REGRESSOR,
        save_path=SAVE_PATH,
        data_size=DATA_SIZE,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        gpu=GPU,
        regressor_activations=REGRESSOR_ACTIVATIONS
    )