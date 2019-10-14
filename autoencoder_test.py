import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
from networks import FourLayerCVAE, PrintLogger
from networks import run_epoch, run_training, dense_net
from networks_test import train_autoencoder, dict_to_batches, load_npz_data
from stl10_test import read_stl_images
from svhn_test import read_svhn_data
import math
import datetime
import os
import csv
import sys

ENCODER_LIST = [
    "lunarlander_experiments/FourLayerCVAE_2019-09-03_10h49.pt",
    "lunarlander_experiments/FourLayerCVAE_2019-09-03_12h47.pt",
    "lunarlander_experiments/FourLayerCVAE_2019-09-03_11h17.pt",
    "lunarlander_experiments/FourLayerCVAE_2019-09-03_12h16.pt",
    "lunarlander_experiments/FourLayerCVAE_2019-09-03_13h45.pt",
    "lunarlander_experiments/FourLayerCVAE_2019-09-03_15h44.pt",
    "lunarlander_experiments/FourLayerCVAE_2019-09-03_14h13.pt",
    "lunarlander_experiments/FourLayerCVAE_2019-09-03_15h12.pt",
    "lunarlander_experiments/FourLayerCVAE_2019-09-03_16h43.pt",
    "lunarlander_experiments/FourLayerCVAE_2019-09-03_17h42.pt",
    "lunarlander_experiments/FourLayerCVAE_2019-09-03_17h11.pt",
    "lunarlander_experiments/FourLayerCVAE_2019-09-03_18h11.pt",
    "lunarlander_experiments/FourLayerCVAE_2019-09-03_19h42.pt",
    "lunarlander_experiments/FourLayerCVAE_2019-09-03_21h42.pt",
    "lunarlander_experiments/FourLayerCVAE_2019-09-03_20h10.pt",
    "lunarlander_experiments/FourLayerCVAE_2019-09-03_21h10.pt",
]
EXPERIMENT = "lunarlander"


if EXPERIMENT == "lunarlander":
    encoder_data = load_npz_data(
        "LunarLander-v2_105000_Dataset.npz",
        90000,
        500,
        [0.8, 0.2]
    )
    encoder_test = load_npz_data(
        "LunarLander-v2_92623_Cleaned.npz",
        90000,
        500,
        [1]
    )
    input_size = (64,64)
    save_path = "lunarlander_experiments"
elif EXPERIMENT == "stl10":
    encoder_data = {
        "imgs" : read_stl_images("stl-10/data/stl10_binary/unlabeled_X.bin")
        }
    encoder_data = dict_to_batches(
        encoder_data, 90000, 500, [0.8, 0.2]
    )
    encoder_test = {
        "imgs" : read_stl_images("stl-10/data/stl10_binary/test_X.bin")
        }
    encoder_test = dict_to_batches(
        encoder_test, len(encoder_test["imgs"]), 500, [1]
    )
    input_size = (96,96)
    save_path = "stl10_experiments"
elif EXPERIMENT == "svhn":
    encoder_data, _  = read_svhn_data("svhn/extra_32x32.mat")
    encoder_data = dict_to_batches(
        {"imgs" : encoder_data}, 90000, 500, [0.8, 0.2]
    )
    input_size = (64,64)
    save_path = "svhn_experiments"

experiment_data = {
        'experiment' : [],
        'encoder_file' : [],
        'model_file' : [],
        'z_dims' : [],
        'perceptual' : [],
        'variational' : [],
        'gamma' : [],
        'l1_loss' : [],
        'mse_loss' : [],
    }

for encoder in ENCODER_LIST:
    model = torch.load(encoder, map_location="cpu")
    variational = model.variational
    gamma = model.gamma
    perceptual = model.perceptual
    z_dims = model.z_dimensions
    model = None
    model, model_file = train_autoencoder(
        data = encoder_data,
        network = encoder,
        epochs = 100,
        gamma = 0,
        variational=variational,
        train_only_decoder = True,
        save_path = save_path,
        gpu = torch.cuda.is_available()
    )
    optimizer = torch.optim.Adam(model.parameters())
    l1 = nn.L1Loss()
    mse = nn.MSELoss()
    
    loss = lambda output, target: [
        l1(output, target),
        mse(output, target)
    ]
    l1_loss, mse_loss = run_epoch(
        network = encoder,
        data = encoder_test[0],
        labels = encoder_test[0],
        loss = loss,
        optimizer = None
    )
    l1_loss = l1_loss/len(encoder_test[0])
    mse_loss = mse_loss/len(encoder_test[0])

    experiment_data['experiment'].append(EXPERIMENT)
    experiment_data['encoder_file'].append(encoder),
    experiment_data['model_file'].append(model_file)
    experiment_data['z_dims'].append(z_dims)
    experiment_data['perceptual'].append(perceptual)
    experiment_data['variational'].append(variational)
    experiment_data['gamma'].append(gamma)
    experiment_data['l1_loss'].append(l1_loss)
    experiment_data['mse_loss'].append(mse_loss)

    with open('autoencoder_test_' + EXPERIMENT + '.csv', 'a+') as f:
        writer = csv.writer(f, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(EXPERIMENT, encoder, model_file, z_dims, perceptual, variational, gamma, l1_loss, mse_loss)
with open('autoencoder_' + EXPERIMENT + '.csv', 'w') as f:
    f.write(str(experiment_data))