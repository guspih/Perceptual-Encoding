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

EXPERIMENTS_PATH = "lunarlander_experiments"
EXPERIMENTS_FILE = "experiments_data.csv"
try:
    os.makedirs(EXPERIMENTS_PATH)
except FileExistsError:
    pass


def run_lunarlander_experiment(
    data_file, encoder_file=None, load_regressor_file=None,
    save_path="lunarlander_experiments", data_size=90000,
    batch_size=1000,splits=[0.4, 0.1, 0.3, 0.1], epochs=20,
    gpu=False, regressor_layers=[2], regressor_activations=[None]
):

    '''
    Trains a regressor on the LunarLander dataset using a CVAE trained on the
    data and a dense network that predicts lander position from the encodings

    Args:
        data_file
        encoder_file (str / None): Path to encoder to use
        load_regressor_file (str / None): Path to regressor to use
        save_path (str): Path to folder where to store trained models
        data_size (int): Number of entries to use from the dataset
        batch_size (int): Size of each batch
        splits ([float]): List of four fractions for different splits
        epochs (int): Number of training epochs
        gpu (bool): Whether to train on the GPU
        regressor_layers ([int]): Layer sizes of regressor if no file provided
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

    train1, validation, train2, test = load_npz_data(
        data_file, data_size, batch_size, splits, gpu=gpu
    )

    if encoder_file is None:
        encoder, encoder_file = train_autoencoder(
            [train1,validation],
            FourLayerCVAE,
            50,
            "LunarLander-Pretraining"
        )
    else:
        encoder = torch.load(encoder_file)

    if load_regressor_file is None:
        regressor = dense_net(
            encoder.z_dimensions,
            regressor_layers,
            regressor_activations
        )
    else:
        regressor = torch.load(load_regressor_file)

    if gpu:
        encoder.cuda()
        regressor.cuda()

    encoder.eval()
    with torch.no_grad():
        print("Generating encodings from images...")
        for split in [train1, train2, validation, test]:
            split["encodings"] = []
            for i, img_batch in enumerate(split["imgs"]):
                if gpu:
                    img_batch = img_batch.cuda()
                code_batch = encoder.encode(img_batch)
                if gpu:
                    code_batch = code_batch.cpu()
                    img_batch = img_batch.cpu()
                split["encodings"].append(code_batch[0])
            new_observations = []
            for i, obs_batch in enumerate(split["observations"]):
                new_observations.append(obs_batch.narrow(1,0,2))
            split["observations"] = new_observations

    optimizer = torch.optim.Adam(regressor.parameters())
    loss_function = torch.nn.MSELoss()
    loss_functions = lambda output, target : [loss_function(output, target)]

    regressor_file = load_regressor_file
    if epochs != 0 or load_regressor_file == None:
        regressor, regressor_file = run_training(
            model = regressor,
            train_data = (train2["encodings"], train2["observations"]),
            val_data = (validation["encodings"], validation["observations"]),
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
        test["encodings"],
        test["observations"],
        test_losses,
        optimizer,
        "Test",
        False
    )

    experiment_data["test_loss"] = losses[0]

    test_accuracy = losses[1]/len(test["encodings"])
    print(
        "Test - Loss {:.5f} - Mean Error {:.5f}".format(
            losses[0]/len(test["encodings"]), test_accuracy
        )
    )

    experiment_data["test_loss"] = losses[0]
    experiment_data["test_mean_error"] = test_accuracy

    with open(EXPERIMENTS_PATH+"/"+EXPERIMENTS_FILE, "a+") as f:
        f.write("Experiment at {}\n".format(experiment_time).replace("_", " "))
        f.write(str(experiment_data))
        f.write("\n\n")

if __name__ == "__main__":
    '''
    If run directly will train a VAE and
    a regressor from the encodings of that VAE

    Parameters:
        DATA_FILE (str): File path to LunarLander-v2 Dataset
        ENCODER_FILE (str / None): File path to autoencoder for encoding
        LOAD_FILE (str / None): File path to regressor for predition
        SAVE_PATH (str): Path to folder where to store trained models
        DATA_SIZE (int): Number of entries to use from the dataset
        BATCH_SIZE (int): Size of each batch
        SPLIT ([float]): List of four fractions for different splits
        EPOCHS (int): Number of training epochs
        GPU (bool): Whether to train on the GPU
        REGRESSOR_LAYERS ([int]): List of layer sizes
        REGRESSOR_ACTIVATIONS ([f()->nn.Module]): List of activation functions
    '''
    
    DATA_FILE = "LunarLander-v2_92623_Cleaned.npz"
    ENCODER_FILE = "autoencoder_checkpoints/CVAE_64x64_2019-06-28_13h58.pt"
    LOAD_FILE = None
    SAVE_PATH = "lunarlander_experiments"
    DATA_SIZE = 90000
    BATCH_SIZE = 1000
    SPLITS = [0.3, 0.1, 0.4, 0.1]
    EPOCHS = 50
    GPU = torch.cuda.is_available()
    REGRESSOR_LAYERS = [2]
    REGRESSOR_ACTIVATIONS = [None]

    run_lunarlander_experiment(
        data_file=DATA_FILE,
        encoder_file=ENCODER_FILE,
        load_regressor_file=LOAD_FILE,
        save_path=SAVE_PATH,
        data_size=DATA_SIZE,
        batch_size=BATCH_SIZE,
        splits=SPLITS,
        epochs=EPOCHS,
        gpu=GPU,
        regressor_layers=REGRESSOR_LAYERS,
        regressor_activations=REGRESSOR_ACTIVATIONS
    )