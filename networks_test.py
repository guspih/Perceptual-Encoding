import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from networks import CVAE_64x64, train_epoch
from skimage import color
import math


DATA_FILE = "LunarLander-v2_105000_Dataset.npz" #An npz containing 64x64x3 images in entry "imgs"
SAVE_FILE = "gamma_exp.pt"  #Where to save the trained model (saved each epoch)
LOAD_FILE = None  #From where to load an already trained model. If None, no model will be loaded

EPOCHS = 50
BATCH_SIZE = 100

Z_DIMS = 32
PERCEPTUAL_LOSS = False
VARIATIONAL = True #Whether the autoencoder is variational
GAMMA = 0.01

data = np.load(DATA_FILE)
data = data["imgs"][:50000]
np.random.shuffle(data)
data = np.array(data, dtype=np.float32)
data = np.transpose(data, (0,3,1,2))
data = np.split(data, data.shape[0]/BATCH_SIZE)
data = [torch.from_numpy(batch) for batch in data]


model = CVAE_64x64()

if not LOAD_FILE is None:
    model = torch.load(LOAD_FILE)
else:
    model = model = CVAE_64x64(z_dimensions=Z_DIMS, variational=VARIATIONAL, gamma=GAMMA, perceptual_loss=PERCEPTUAL_LOSS)
optimizer = torch.optim.Adam(model.parameters())

def show(imgs, block=False, heading="Figure"):
    #img.reshape(3,64,64)
    plt.close()
    #plt.clf()
    fig=plt.figure(figsize=(8, 8))
    fig.canvas.set_window_title(heading)
    for i, img in enumerate(imgs):
        npimg = img[0].detach().numpy()
        npimg = np.transpose(npimg, (1, 2, 0))
        fig.add_subplot(len(imgs), 1, i+1)
        plt.imshow(npimg)#, interpolation='nearest')
    plt.show(block=block)
    plt.pause(5)

for epoch in range(EPOCHS):
    np.random.shuffle(data)
    epoch_loss = train_epoch(model, data, data, model.loss, optimizer, "Epoch {}".format(epoch))
    print("EPOCH: {}, LOSS: {}".format(epoch, epoch_loss))
    
    torch.save(model, SAVE_FILE)

    r = random.choice(range(data[0].size(0)))
    img1 = torch.index_select(data[0], 0, torch.tensor([r]))
    mu, logvar = model.encode(img1)
    z = model.sample(mu,logvar)
    img2 = model.decode(z)
    show([img1,img2], heading="EPOCH {}".format(epoch))

for d in range(len(data)):
    r = random.choice(range(data[d].size(0)))
    img1 = torch.index_select(data[d], 0, torch.tensor([r]))
    mu, logvar = model.encode(img1)
    z = model.sample(mu,logvar)
    img2 = model.decode(z)
    show([img1,img2], block=True)