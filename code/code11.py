#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import time  # for sleep
import IPython.display as ipd  # for display and clear_output
from IPython.display import display, clear_output  # for the following animation
import os
import copy
import signal
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
import optimizers as opt
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib notebook  # We don't need the interaction with our plots in this notebook

import pickle
import gzip

with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

Xtrain = train_set[0]
Ttrain = train_set[1].reshape(-1, 1)

Xval = valid_set[0]
Tval = valid_set[1].reshape(-1, 1)

Xtest = test_set[0]
Ttest = test_set[1].reshape(-1, 1)

print(Xtrain.shape, Ttrain.shape,  Xval.shape, Tval.shape,  Xtest.shape, Ttest.shape)

three = Xtrain[7, :].reshape(28, 28)

def draw_image(image, label):
    plt.imshow(image, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title(label)

plt.figure()
draw_image(three, 3)

def draw_neg_image(image, label):
    plt.imshow(-image, cmap='gray', vmin=-1, vmax=0)  # <-- New part
    plt.xticks([])
    plt.yticks([])
    plt.title(label)
    
draw_neg_image(three, 3)

patches = []
for row in range(0, 28, 7):
    for col in range(0, 28, 7):
        patches.append(three[row:row + 7, col:col + 7])
len(patches)

plt.figure(figsize=(8, 8))
ploti = 0
for patch in patches:
    ploti += 1
    plt.subplot(4, 4, ploti)
    draw_neg_image(patch, '')
plt.tight_layout()



plt.figure(figsize=(8, 8))
ploti = 0
for patch in patches:
    ploti += 1
    plt.subplot(4, 4, ploti)
    draw_neg_image(patch, '')
plt.tight_layout()

patches = []
for row in range(0, 28, 2):
    for col in range(0, 28, 2):
        patches.append(three[row:row + 7, col:col + 7])
len(patches)

np.sqrt(len(patches))

n_plot_rows = int(np.sqrt(len(patches)))
plt.figure(figsize=(8, 8))
ploti = 0
for patch in patches:
    ploti += 1
    plt.subplot(n_plot_rows, n_plot_rows, ploti)
    draw_neg_image(patch, '')
patches = []
for row in range(0, 28, 2):
    for col in range(0, 28, 2):
        if row + 7 < 28 and col + 7 < 28:
            patches.append(three[row:row + 7, col:col + 7])
len(patches)

n_plot_rows = int(np.sqrt(len(patches)))
plt.figure(figsize=(8, 8))
ploti = 0
for patch in patches:
    ploti += 1
    plt.subplot(n_plot_rows, n_plot_rows, ploti)
    draw_neg_image(patch, '')

three[0:4, 0:4] = 1.0

n_plot_rows = int(np.sqrt(len(patches)))
plt.figure(figsize=(8, 8))
ploti = 0
for patch in patches:
    ploti += 1
    plt.subplot(n_plot_rows, n_plot_rows, ploti)
    draw_neg_image(patch, '')

# =============================================================================
# Weight matrix as kernel or filter
# =============================================================================

weights = np.array([[-1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1,  1],
                    [-1, -1, -1, -1, -1,  1,  1],
                    [-1, -1, -1, -1,  1,  1,  1],
                    [-1, -1, -1,  1,  1,  1,  1],
                    [-1, -1,  1,  1,  1,  1,  1],
                    [-1,  1,  1,  1,  1,  1,  1]])
weights

plt.imshow(weights, cmap='gray')
plt.colorbar();

new_image = []
for patch in patches:
    new_image.append( np.sum(patch * weights) )
new_image = np.array(new_image)
new_image

new_image_dim = int(np.sqrt(len(new_image)))
new_image_dim

new_image = new_image.reshape(new_image_dim, new_image_dim)
draw_image(new_image, '')
plt.colorbar();

patches_array = np.array(patches)
patches_array.shape

new_image = patches_array.reshape(121, -1) @ weights.reshape(-1, 1)
new_image.shape

new_image = new_image.reshape(new_image_dim, new_image_dim)
draw_image(new_image, '')
plt.colorbar();

import numpy.lib.stride_tricks

def make_patches(X, patch_size, stride=1):
    '''X: n_samples x n_pixels  (flattened square images)'''
    X = np.ascontiguousarray(X)  # make sure X values are contiguous in memory
    
    n_samples = X.shape[0]
    image_size = int(np.sqrt(X.shape[1]))
    n_patches = (image_size - patch_size ) // stride + 1
    
    nb = X.itemsize  # number of bytes each value

    new_shape = [n_samples, 
                 n_patches,  # number of rows of patches
                 n_patches,  # number of columns of patches
                 patch_size, # number of rows of pixels in each patch
                 patch_size] # number of columns of pixels in each patch
    
    new_strides = [image_size * image_size * nb,  # nuber of bytes to next image (sample)
                   image_size * stride * nb,      # number of bytes to start of next patch in next row
                   stride * nb,                   # number of bytes to start of next patch in next column
                   image_size * nb,               # number of bytes to pixel in next row of patch
                   nb]                            # number of bytes to pixel in next column of patch
    
    X = np.lib.stride_tricks.as_strided(X, shape=new_shape, strides=new_strides)
    
    # Reshape the set of patches to two-dimensional matrix, of shape  N x P x S,
    #   N is number of samples,  P is number of patches,  S is number of pixels per patch
    X = X.reshape(n_samples, n_patches * n_patches, patch_size * patch_size)
    
    return X

patches = make_patches(Xtrain[:2], 7, 2)

n_plot_rows = int(np.sqrt(patches.shape[1]))
patch_size = int(np.sqrt(patches.shape[2]))

for patchi in range(patches.shape[0]):
    print(Ttrain[patchi])
    plt.figure(figsize=(8, 8))
    ploti = 0
    for patch in patches[patchi, :, :]:
        ploti += 1
        plt.subplot(n_plot_rows, n_plot_rows, ploti)
        draw_neg_image(patch.reshape(patch_size, patch_size), '')

plt.imshow(weights, cmap='gray');

weights_flipped = np.flipud(weights)
plt.imshow(weights_flipped, cmap='gray');

weights = weights.reshape(49, 1)
weights.shape

weights_flipped = weights_flipped.reshape(49, 1)
weights_flipped.shape

weights_both = np.hstack((weights, weights_flipped))

plt.subplot(1, 2, 1)
plt.imshow(weights_both[:, 0].reshape(7, 7), cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(weights_both[:, 1].reshape(7, 7), cmap='gray');

patches.shape, weights_both.shape

output = patches @ weights_both
output.shape

for i in range(2):
    plt.figure()
    for j in range(2):
        plt.subplot(1, 2, j+1)
        output_image = output[i, :, j].reshape(11, 11)
        draw_image(output_image, '')
















