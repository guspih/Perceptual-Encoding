import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from networks import CVAE_64x64, train_epoch
from skimage import color
import math

def show(imgs, block=False, heading="Figure"):
    '''
    Paints a column of torch images

    Args:
        imgs ([3darray]): Array of images in shape (channels, width, height)
        block (bool): Whether the image should interupt program flow
        heading (str)): The heading to put on the image
    '''
    plt.close()
    fig=plt.figure(figsize=(8, 8))
    fig.canvas.set_window_title(heading)
    for i, img in enumerate(imgs):
        img = img[0].detach().permute(1,2,0)
        fig.add_subplot(len(imgs), 1, i+1)
        plt.imshow(img)
    plt.show(block=block)
    plt.pause(5)

def show_recreation(data, epoch=0, batch=0, block=False):
    '''
    Shows a random image and the encoders attempted recreation

    Args:
        data ([tensor]): List of image batches
        epoch (int): Epoch to be printed on the heading
        batch (int): Batch of data to sample the image from
        block (bool): Whether to stop execution until user closes image
    '''
    r = random.choice(range(data[batch].size(0)))
    img1 = torch.index_select(data[batch], 0, torch.tensor([r]))
    mu, logvar = model.encode(img1)
    z = model.sample(mu,logvar)
    img2 = model.decode(z)
    show([img1,img2], block=block, heading="Random image from: Epoch {}, Batch {}".format(epoch, batch))

def load_image_data(data_file, data_size, batch_size, shuffle=True, gpu=False):
    '''
    Loads images from key 'imgs' from an .npz file

    Args:
        data_file (str): Path to file with data
        data_size (int): Amount of images to load
        batch_size (int): How much data will be stored in each batch
        shuffle (bool): Whether to shuffle the data
        gpu (bool): Whether data will be used on the gpu

    Returns: ([tensor])
    '''
    assert data_size % batch_size == 0, \
        "data_size must be divisble by batch_size"

    data = np.load(data_file)
    data = data["imgs"][:data_size]
    if shuffle:
        np.random.shuffle(data)
    data = np.array(data, dtype=np.float32)
    data = np.transpose(data, (0,3,1,2))
    data = np.split(data, data.shape[0]/batch_size)
    data = [torch.from_numpy(batch) for batch in data]
    if gpu:
        data = [batch.cuda() for batch in data]
    return data



DATA_FILE = "LunarLander-v2_105000_Dataset.npz" #An npz containing 64x64x3 images in entry "imgs"
SAVE_FILE = "network_test.pt"  #Where to save the trained model (saved each epoch)
LOAD_FILE = "gamma_exp.pt"  #From where to load an already trained model. If None, no model will be loaded

EPOCHS = 0
BATCH_SIZE = 100

Z_DIMS = 32
PERCEPTUAL_LOSS = False
VARIATIONAL = True #Whether the autoencoder is variational
GAMMA = 0.01


data = load_image_data(DATA_FILE, 50000, BATCH_SIZE, True, gpu=False)

if not LOAD_FILE is None:
    model = torch.load(LOAD_FILE)
else:
    model = model = CVAE_64x64(
        z_dimensions=Z_DIMS,
        variational=VARIATIONAL,
        gamma=GAMMA,
        perceptual_loss=PERCEPTUAL_LOSS
    )

optimizer = torch.optim.Adam(model.parameters())

for epoch in range(EPOCHS):
    np.random.shuffle(data)
    epoch_loss = train_epoch(model, data, data, model.loss, optimizer, "Epoch {}".format(epoch))
    print("EPOCH: {}, LOSS: {}".format(epoch, epoch_loss))
    
    torch.save(model, SAVE_FILE)

    show_recreation(data, epoch=epoch, batch=0, block=False)

for batch_id in range(len(data)):
    show_recreation(data, epoch=EPOCHS, batch=batch_id, block=True)