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


EXPERIMENT = "stl10"

if EXPERIMENT == "lunarlander":
    encoder_list = [
        ("lunarlander_experiments/FourLayerCVAE_2019-09-03_10h49.pt", "lunarlander_experiments/FourLayerCVAE_2019-10-15_16h29.pt"),
        ("lunarlander_experiments/FourLayerCVAE_2019-09-03_12h47.pt", "lunarlander_experiments/FourLayerCVAE_2019-10-15_16h50.pt"),
        ("lunarlander_experiments/FourLayerCVAE_2019-09-03_11h17.pt", "lunarlander_experiments/FourLayerCVAE_2019-10-15_17h12.pt"),
        ("lunarlander_experiments/FourLayerCVAE_2019-09-03_12h16.pt", "lunarlander_experiments/FourLayerCVAE_2019-10-15_17h33.pt"),
        ("lunarlander_experiments/FourLayerCVAE_2019-09-03_13h45.pt", "lunarlander_experiments/FourLayerCVAE_2019-10-15_17h54.pt"),
        ("lunarlander_experiments/FourLayerCVAE_2019-09-03_15h44.pt", "lunarlander_experiments/FourLayerCVAE_2019-10-15_18h16.pt"),
        ("lunarlander_experiments/FourLayerCVAE_2019-09-03_14h13.pt", "lunarlander_experiments/FourLayerCVAE_2019-10-15_18h37.pt"),
        ("lunarlander_experiments/FourLayerCVAE_2019-09-03_15h12.pt", "lunarlander_experiments/FourLayerCVAE_2019-10-15_18h58.pt"),
        ("lunarlander_experiments/FourLayerCVAE_2019-09-03_16h43.pt", "lunarlander_experiments/FourLayerCVAE_2019-10-15_19h20.pt"),
        ("lunarlander_experiments/FourLayerCVAE_2019-09-03_17h42.pt", "lunarlander_experiments/FourLayerCVAE_2019-10-15_19h41.pt"),
        ("lunarlander_experiments/FourLayerCVAE_2019-09-03_17h11.pt", "lunarlander_experiments/FourLayerCVAE_2019-10-15_20h02.pt"),
        ("lunarlander_experiments/FourLayerCVAE_2019-09-03_18h11.pt", "lunarlander_experiments/FourLayerCVAE_2019-10-15_20h24.pt"),
        ("lunarlander_experiments/FourLayerCVAE_2019-09-03_19h42.pt", "lunarlander_experiments/FourLayerCVAE_2019-10-15_20h45.pt"),
        ("lunarlander_experiments/FourLayerCVAE_2019-09-03_21h42.pt", "lunarlander_experiments/FourLayerCVAE_2019-10-15_21h06.pt"),
        ("lunarlander_experiments/FourLayerCVAE_2019-09-03_20h10.pt", "lunarlander_experiments/FourLayerCVAE_2019-10-15_21h28.pt"),
        ("lunarlander_experiments/FourLayerCVAE_2019-09-03_21h10.pt", "lunarlander_experiments/FourLayerCVAE_2019-10-15_21h49.pt")
    ]
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
    encoder_list = [
        ("stl10_experiments/FourLayerCVAE_2019-09-11_11h20.pt",""),
        ("stl10_experiments/FourLayerCVAE_2019-09-11_15h28.pt",""),
        ("stl10_experiments/FourLayerCVAE_2019-09-11_12h12.pt",""),
        ("stl10_experiments/FourLayerCVAE_2019-09-11_14h04.pt",""),
        ("stl10_experiments/FourLayerCVAE_2019-09-11_17h18.pt",""),
        ("stl10_experiments/FourLayerCVAE_2019-09-11_20h58.pt",""),
        ("stl10_experiments/FourLayerCVAE_2019-09-11_18h09.pt",""),
        ("stl10_experiments/FourLayerCVAE_2019-09-11_20h00.pt",""),
        ("stl10_experiments/FourLayerCVAE_2019-09-11_22h48.pt",""),
        ("stl10_experiments/FourLayerCVAE_2019-09-12_00h37.pt",""),
        ("stl10_experiments/FourLayerCVAE_2019-09-11_23h39.pt",""),
        ("stl10_experiments/FourLayerCVAE_2019-09-12_03h19.pt",""),
        ("stl10_experiments/FourLayerCVAE_2019-09-12_04h18.pt",""),
        ("stl10_experiments/FourLayerCVAE_2019-09-12_06h08.pt",""),
        ("stl10_experiments/FourLayerCVAE_2019-09-12_05h10.pt",""),
        ("stl10_experiments/FourLayerCVAE_2019-09-12_08h53.pt","")
    ]
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
    encoder_list = [
        ("svhn_experiments/FourLayerCVAE_2019-10-01_14h14.pt",""),
        ("svhn_experiments/FourLayerCVAE_2019-10-01_15h15.pt",""),
        ("svhn_experiments/FourLayerCVAE_2019-10-01_14h43.pt",""),
        ("svhn_experiments/FourLayerCVAE_2019-10-01_15h44.pt",""),
        ("svhn_experiments/FourLayerCVAE_2019-10-01_17h18.pt",""),
        ("svhn_experiments/FourLayerCVAE_2019-10-01_18h19.pt",""),
        ("svhn_experiments/FourLayerCVAE_2019-10-01_17h46.pt",""),
        ("svhn_experiments/FourLayerCVAE_2019-10-01_18h48.pt",""),
        ("svhn_experiments/FourLayerCVAE_2019-10-01_20h22.pt",""),
        ("svhn_experiments/FourLayerCVAE_2019-10-01_21h24.pt",""),
        ("svhn_experiments/FourLayerCVAE_2019-10-01_20h51.pt",""),
        ("svhn_experiments/FourLayerCVAE_2019-10-01_21h54.pt",""),
    ]    
    encoder_data, _ = read_svhn_data("svhn/extra_32x32.mat")
    encoder_data = dict_to_batches(
        {"imgs" : encoder_data}, 90000, 500, [0.8, 0.2]
    )
    encoder_test, _ = read_svhn_data("svhn/test_32x32.mat")
    encoder_test = {"imgs" : encoder_test}
    encoder_test, = dict_to_batches(
        encoder_test,
        data_size=len(encoder_test["imgs"]),
        batch_size=500,
        split_distribution=[1],
        uneven_batches=True
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

for encoder, _ in encoder_list:
    model = torch.load(encoder, map_location="cpu")
    variational = model.variational
    gamma = model.gamma
    perceptual = model.perceptual_loss
    z_dims = model.z_dimensions
    model = None
    model, model_file = train_autoencoder(
        data = encoder_data,
        network = encoder,
        epochs = 100,
        gamma = 0,
        variational=False,
        train_only_decoder = True,
        save_path = save_path,
        gpu = torch.cuda.is_available()
    )

    optimizer = torch.optim.Adam(model.parameters())
    l1 = nn.L1Loss()
    mse = nn.MSELoss()
    loss = lambda output, target: [
        l1(output[0], target),
        mse(output[0], target)
    ]
    l1_loss, mse_loss = run_epoch(
        network = model,
        data = encoder_test[0]["imgs"],
        labels = encoder_test[0]["imgs"],
        loss = loss,
        optimizer = optimizer
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
        writer.writerow([EXPERIMENT, encoder, model_file, z_dims, perceptual, variational, gamma, l1_loss, mse_loss])
with open('autoencoder_' + EXPERIMENT + '.csv', 'w') as f:
    f.write(str(experiment_data))