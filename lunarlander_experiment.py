import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import math
import datetime
import csv
import os
import sys

from networks import CVAE_64x64, dense_net, run_epoch, PrintLogger
from networks_test import load_npz_data, train_autoencoder

DATA_FILE = "LunarLander-v2_92623_Cleaned.npz"
ENCODER_FILE = "autoencoder_checkpoints/CVAE_64x64_2019-06-27_14h53.pt"
LOAD_FILE = None
SAVE_PATH = "lunarlander_checkpoints"
DATA_SIZE = 90000
BATCH_SIZE = 1000
SPLITS = [0.3, 0.1, 0.4, 0.1]
EPOCHS = 20
GPU = torch.cuda.is_available()
REGRESSOR_LAYERS = [2]
REGRESSOR_ACTIVATIONS = [None]

# def run_lunarlander_experiment(
#     data_file, encoder_file=None, load_file=None, save_path="lunarlander_experiments",


# ):

log_file = "log_"+datetime.datetime.now().strftime("%Y-%m-%d_%Hh%M")+".log"
if SAVE_PATH != "":
    if not os.path.isdir(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    log_file = SAVE_PATH+"/"+log_file
PrintLogger([log_file])

print("Starting LunarLander-v2 regression experiment...")
print(
    "Dataset is {}\nEncoder is {}\nRegressor is {}".format(
        DATA_FILE,
        ENCODER_FILE if ENCODER_FILE != None else "going to be trained",
        LOAD_FILE if LOAD_FILE != None else "going to be trained"
    )
)

train1, validation, train2, test = load_npz_data(
    DATA_FILE, DATA_SIZE, BATCH_SIZE, SPLITS, gpu=GPU
)

if ENCODER_FILE is None:
    encoder = train_autoencoder(
        [train1,validation],
        CVAE_64x64,
        50,
        "LunarLander-Pretraining"
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

if GPU:
    encoder.cuda()
    regressor.cuda()

save_file = "LunarLander_Regressor_{}_{}".format(
    encoder.z_dimensions,
    datetime.datetime.now().strftime("%Y-%m-%d_%Hh%M")
).replace(".","-") + ".pt"
if SAVE_PATH != "":
    save_file = SAVE_PATH+"/"+save_file

encoder.eval()
with torch.no_grad():
    print("Generating encodings from images...")
    for split in [train1, train2, validation, test]:
        split["encodings"] = []
        for i, img_batch in enumerate(split["imgs"]):
            code_batch = encoder.encode(img_batch)
            if GPU:
                code_batch[0].cuda()
                code_batch[1].cuda()
            split["encodings"].append(code_batch[0])
        new_observations = []
        for i, obs_batch in enumerate(split["observations"]):
            new_observations.append(obs_batch.narrow(1,0,2))
        split["observations"] = new_observations

optimizer = torch.optim.Adam(regressor.parameters())
loss_function = torch.nn.MSELoss()
loss_functions = lambda output, target : [loss_function(output, target)]


print(
    "Starting regressor training. Best checkpoint stored in {}".format(
        save_file
    )
)

best_validation_loss = float("inf")
regressor.train()
for epoch in range(1,EPOCHS+1):
    # epoch_training_loss = 0
    # for i, (source, obs) in enumerate(
    #     zip(train2["encodings"], train2["observations"])
    # ):
    #     target = obs.narrow(1,0,2)
    #     optimizer.zero_grad()
    #     output = regressor(source[0])
    #     loss = loss_function(output, target)
    #     epoch_training_loss = epoch_training_loss+loss.item()
    #     loss.backward()
    #     optimizer.step()
    #     print(
    #         "\rEpoch {} - Training - [{}/{}]".format(
    #             epoch+1, i+1, len(train2["encodings"])
    #         ),
    #         end=""
    #     )

    epoch_training_loss = run_epoch(
        regressor,
        train2["encodings"],
        train2["observations"],
        loss_functions,
        optimizer,
        "Train " + str(epoch),
        True
    )

    # epoch_validation_loss = 0
    # with torch.no_grad():
    #     for i, (source, obs) in enumerate(
    #         zip(validation["encodings"], validation["observations"])
    #     ):
    #         target = obs.narrow(1,0,2)
    #         output = regressor(source[0])
    #         loss = loss_function(output, target)
    #         epoch_validation_loss = epoch_validation_loss+loss.item()
    #         print(
    #             "\rEpoch {} - Validation - [{}/{}]".format(
    #                 epoch+1, i+1, len(validation["encodings"])
    #             ),
    #             end=""
    #         )
    epoch_validation_loss = run_epoch(
        regressor,
        validation["encodings"],
        validation["observations"],
        loss_functions,
        optimizer,
        "Validation " + str(epoch),
        False
    )
    if epoch_validation_loss[0] < best_validation_loss:
        best_validation_loss = epoch_validation_loss[0]
        torch.save(regressor.cpu(), save_file)
        if GPU:
            regressor.cuda()

# with torch.no_grad():
#     for i, (source, obs) in enumerate(
#         zip(test["encodings"], test["observations"])
#     ):
#         target = obs.narrow(1,0,2)
#         output = regressor(source[0])
#         loss = loss_function(output, target)
#         accuracy = torch.mean(torch.abs(output-target))
#         test_loss = test_loss+loss.item()
#         test_accuracy = test_accuracy+accuracy.item()

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

test_accuracy = losses[1]/len(test["encodings"])
print(
    "Test - Loss {:.5f} - Mean Error {:.5f}".format(
        losses[0]/len(test["encodings"]), test_accuracy
    )
)