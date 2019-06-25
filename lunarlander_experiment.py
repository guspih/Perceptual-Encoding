import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from networks import CVAE_64x64, dense_net, run_epoch
from networks_test import load_npz_data, train_autoencoder
import math
import datetime
import csv

DATA_FILE = "LunarLander-v2_105000_Dataset.npz"
ENCODER_FILE = None
LOAD_FILE = None
DATA_SIZE = 90000
BATCH_SIZE = 1000
SPLITS = [0.3, 0.1, 0.4, 0.1]
EPCOHS = 100
GPU = torch.cuda.is_available()
REGRESSOR_LAYERS = [2]
REGRESSOR_ACTIVATIONS = [None]

train1, validation, train2, test = load_npz_data(
    DATA_FILE, DATA_SIZE, BATCH_SIZE, SPLITS, gpu=GPU
)

if ENCODER_FILE is None:
    encoder = train_autoencoder(
        [train1,validation],
        CVAE_64x64,
        50,
        "LunarLander-Pretraining",
        gpu=GPU
    )
else:
    encoder = torch.load(ENCODER_FILE)

if LOAD_FILE is None:
    regressor = dense_net(
        encoder.z_dimensions,
        REGRESSOR_LAYERS,
        REGRESSOR_ACTIVATIONS
    )
else:
    regressor = torch.load(LOAD_FILE)

if gpu:
    encoder.gpu()
    regressor.gpu()

for split in [train1, train2, validation, test]:
    split["encodings"] = []
    for i, img_batch in enumerate(split["imgs"]):
        code_batch = encoder.encode(img_batch)
        split["encodings"].append(code_batch)

optimizer = torch.optim.Adam(regressor.parameters())
loss_function = torch.nn.MSELoss()
save_file = "LunarLander_Regressor_{}_{}.pt".format(
    encoder.z_dimensions,
    datetime.datetime.now().strftime("%Y-%m-%d_%Hh%M")
).replace(".","-")

best_validation_loss = float("inf")
regressor.train()
for epoch in range(EPCOHS):
    epoch_training_loss = 0
    for i, (source, target) in enumerate(
        zip(train2["imgs"], train2["observations"][:,:,:2])
    ):
        optimizer.zero_grad()
        output = regressor(source)
        loss = loss_function(output, target)
        epoch_training_loss = epoch_training_loss+loss.item()
        loss.backward()
        optimizer.step()   

    epoch_validation_loss = 0
    for i, (source, target) in enumerate(
        zip(validation["imgs"], validation["observations"][:,:,:2])
    ):
        optimizer.zero_grad()
        output = regressor(source)
        loss = loss_function(output, target)
        epoch_validation_loss = epoch_validation_loss+loss.item()
        loss.backward()
    if epoch_validation_loss < best_validation_loss:
        best_validation_loss = epoch_validation_loss
        torch.save(regressor.cpu(), save_file)
    
test_loss = 0
for i, (source, target) in enumerate(
    zip(test["imgs"], test["observations"][:,:,:2])
):
    optimizer.zero_grad()
    output = regressor(source)
    loss = loss_function(output, target)
    test_loss = test_loss+loss.item()
    loss.backward()

print(test_loss)