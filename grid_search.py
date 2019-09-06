import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
import math
import datetime
import os
import csv
import sys
from itertools import combinations_with_replacement, product
from networks import FourLayerCVAE, PrintLogger
from networks import run_epoch, run_training, dense_net
from networks_test import train_autoencoder, dict_to_batches, load_npz_data
from lunarlander_experiment import run_lunarlander_experiment
from stl10_test import stl10_experiment, read_stl_images, read_stl_labels

def generate_dense_architectures(hidden_sizes, hidden_nrs):
    '''
    Given acceptable sizes for hidden layers and acceptable number of layers,
    generates all feasible architectures to test.

    Args:
        hidden_sizes ([int]): List of acceptable sizes of the hidden layers
        hidden_nrs ([int]): List of acceptable number of layers
    
    Returns ([[int]]): List of architectures consisting of list of layer sizes
    '''
    archs = []
    hidden_sizes.sort(reverse=True)
    for hidden_nr in hidden_nrs:
        archs = archs + list(combinations_with_replacement(hidden_sizes, hidden_nr))
    return [list(arch) for arch in archs]

def grid_search(
    experiment="lunarlander", z_dims=[64,128,256,512], gammas=[0,0.001,0.01],
    predictor_hidden_sizes=[32,64,128], predictor_hidden_nrs=[0,1,2],
    predictor_hidden_funcs=[nn.LeakyReLU, nn.Sigmoid],
    predictor_out_func=[None] 
):
    '''
    Does grid search over many different parameters of encoders and predictors
    for the different experiments. Only the ones with best validation will be
    tested.

    Args:
        experiment (str): What experiment to gridsearch (lunarlander, stl10)
        z_dims ([int]): Values of z_dimensions to test
        gammas ([float]): Values of gamma to test. 0 implies non-variational
        predictor_hidden_sizes ([int]): Sizes of the hidden layers to test
        predictor_hidden_nrs ([int]): Number of hidden layers to test
        predictor_hidden_funcs ([f()->nn.Module]): Activations of hidden layers
        predictor_out_func ([f()->nn.Module]): Activations of final layer
    '''

    # Set variables
    data_size = 90000
    batch_size = 500
    encoder_epochs = 100
    predictor_epochs = 1000
    gpu = torch.cuda.is_available()

    # Load experiment relevant information
    if experiment == "lunarlander":
        encoder_data = load_npz_data(
            "LunarLander-v2_105000_Dataset.npz",
            data_size,
            batch_size,
            [0.8, 0.2]
        )
        experiment_function = run_lunarlander_experiment
        input_size = (64,64)
        save_path = "lunarlander_experiments"
        output_size = 2
    elif experiment == "stl10":
        encoder_data = {
            "imgs" : read_stl_images("stl-10/data/stl10_binary/unlabeled_X.bin")
            }
        encoder_data = dict_to_batches(
            encoder_data, data_size, batch_size, [0.8, 0.2]
        )
        experiment_function = stl10_experiment
        input_size = (96,96)
        save_path = "stl10_experiments"
        output_size = 10

    # Train and store paths to all encoder models 
    encoders_etc = []  
    for z_dim, gamma, perceptual in product(
        z_dims, gammas, [False, True]
    ):
        variational = gamma != 0
        _, model_path = train_autoencoder(
            encoder_data, FourLayerCVAE, encoder_epochs, input_size,
            z_dim, variational, gamma, perceptual,
            gpu, False, save_path, False
        )
        encoders_etc.append((model_path, z_dim, gamma, perceptual, variational))
    
    # Dereference unnecessary data to free up space
    encoder_data = None

    # Run all experiment permutation with all different encoders
    results = {}
    architectures = generate_dense_architectures(
        predictor_hidden_sizes,
        predictor_hidden_nrs
    )
    for encoder_etc, architecture, hidden_func, out_func in product(
        encoders_etc, architectures, predictor_hidden_funcs, predictor_out_func
    ):
        encoder_path, z_dim, gamma, perceptual, variational = encoder_etc
        architecture = architecture.copy()
        architecture.append(output_size)
        act_functs = [hidden_func]*(len(architecture)-1) + [out_func]
        experiment_data = experiment_function(
            encoder_path, architecture, save_path, data_size, batch_size,
            predictor_epochs, gpu, act_functs
        )

        results[(
            encoder_path,
            z_dim,
            gamma,
            perceptual,
            variational,
            tuple(architecture),
            hidden_func, 
            out_func
        )] = experiment_data

    #Store the results

    with open("grid_search_" + experiment + ".csv", "a+") as f:
        f.write(str(results))

if __name__ == "__main__":

    grid_search(experiment="stl10", predictor_out_func=[None, nn.Softmax])