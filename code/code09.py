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


n = 500
x1 = np.linspace(5, 20, n) + np.random.uniform(-2, 2, n)
y1 = ((20-12.5)**2-(x1-12.5)**2) / (20-12.5)**2 * 10 + 14 + np.random.uniform(-2, 2, n)
x2 = np.linspace(10, 25, n) + np.random.uniform(-2, 2, n)
y2 = ((x2-17.5)**2) / (25-17.5)**2 * 10 + 5.5 + np.random.uniform(-2, 2, n)
angles = np.linspace(0, 2*np.pi, n)
x3 = np.cos(angles) * 15 + 15 + np.random.uniform(-2, 2, n)
y3 = np.sin(angles) * 15 + 15 + np.random.uniform(-2, 2, n)
X =  np.vstack((np.hstack((x1, x2, x3)),  np.hstack((y1, y2, y3)))).T
T = np.repeat(range(1, 4), n).reshape((-1, 1))
colors = ['blue', 'red', 'green']
plt.figure(figsize=(6, 6))
for c in range(1, 4):
    mask = (T == c).flatten()
    plt.plot(X[mask, 0], X[mask, 1], 'o', markersize=6,  alpha=0.5,  color=colors[c-1])
    
import neuralnetworks as nn
import mpl_toolkits.mplot3d as plt3
from matplotlib import cm

# =============================================================================
# nHidden = [5]
# nnet = nn.NeuralNetworkClassifier(2, nHidden, 3) # 3 classes, will actually make 2-unit output layer
# nnet.train(X, T, n_epochs=5000,  method='scg')
# # nnet.train(X, T, n_epochs=5000,  method='sgd', learning_rate=0.1)
# 
# xs = np.linspace(0, 30, 40)
# x, y = np.meshgrid(xs, xs)
# Xtest = np.vstack((x.flat, y.flat)).T
# predTest, probs = nnet.use(Xtest)
# 
# plt.figure(figsize=(10, 10))
# 
# plt.subplot(2, 2, 1)
# plt.plot(nnet.error_trace)
# plt.xlabel("Epochs")
# plt.ylabel("Likelihood")
# 
# # plt.subplot(2, 2, 3)
# # nnet.draw()
# 
# colors = ['red', 'green', 'blue']
# plt.subplot(2, 2, 2)
# 
# for c in range(1, 4):
#     mask = (T == c).flatten()
#     plt.plot(X[mask, 0], X[mask, 1], 'o', markersize=6,  alpha=0.5,  color=colors[c-1])
# 
# plt.subplot(2, 2, 4)
# plt.contourf(Xtest[:, 0].reshape((40, 40)), Xtest[:, 1].reshape((40, 40)),  predTest.reshape((40, 40)), 
#              levels = [0.5, 1.99, 2.01, 3.5],  #    levels=(0.5, 1.5, 2.5, 3.5), 
#              colors=('red', 'green', 'blue'));
# =============================================================================
    
    
# =============================================================================
# fig = plt.figure(figsize=(6, 20))
# ls = LightSource(azdeg=30,  altdeg=10)
# white = np.ones((x.shape[0],  x.shape[1],  3))
# red = white * np.array([1, 0.2, 0.2])
# green = white * np.array([0.2, 1, 0.2])
# blue = white * np.array([0.4, 0.4, 1])
# colors = [red,  green,  blue]
# 
# for c in range(3):
#     ax = fig.add_subplot(3, 1, c+1, projection='3d')
#     ax.view_init(azim = 180+40, elev = 60)
#     Z = probs[:,  c].reshape(x.shape)
#     rgb = ls.shade_rgb(colors[c],  Z,  vert_exag=0.1)
#     ax.plot_surface(x, y, Z, 
#                     rstride=1, cstride=1, linewidth=0,  antialiased=False, 
#                     shade=True,  facecolors=rgb)
#     ax.set_zlabel(r"$p(C="+str(c+1)+"|x)$")
#     
# =============================================================================


#%%
# =============================================================================
# Accelerometer Data
# =============================================================================
data = np.load('accelerometers.npy')
X = data[:, 1:]
T = data[:, 0:1]
X.shape,  T.shape

def generate_k_fold_cross_validation_sets(X, T, n_folds, shuffle=True):

    if shuffle:
        # Randomly order X and T
        randorder = np.arange(X.shape[0])
        np.random.shuffle(randorder)
        X = X[randorder, :]
        T = T[randorder, :]

    # Partition X and T into folds
    n_samples = X.shape[0]
    n_per_fold = round(n_samples / n_folds)
    n_last_fold = n_samples - n_per_fold * (n_folds - 1)

    folds = []
    start = 0
    for foldi in range(n_folds-1):
        folds.append( (X[start:start + n_per_fold, :], T[start:start + n_per_fold, :]) )
        start += n_per_fold
    folds.append( (X[start:, :], T[start:, :]) )

    # Yield k(k-1) assignments of Xtrain, Train, Xvalidate, Tvalidate, Xtest, Ttest

    for validation_i in range(n_folds):
        for test_i in range(n_folds):
            if test_i == validation_i:
                continue

            train_i = np.setdiff1d(range(n_folds), [validation_i, test_i])

            Xvalidate, Tvalidate = folds[validation_i]
            Xtest, Ttest = folds[test_i]
            if len(train_i) > 1:
                Xtrain = np.vstack([folds[i][0] for i in train_i])
                Ttrain = np.vstack([folds[i][1] for i in train_i])
            else:
                Xtrain, Ttrain = folds[train_i[0]]

            yield Xtrain, Ttrain, Xvalidate, Tvalidate, Xtest, Ttest

def times2():
    for i in range(10):
        yield i * 2

z = times2()
z

Xtrain,  Ttrain,  Xval, Tval, Xtest,  Ttest = next(generate_k_fold_cross_validation_sets(X,  T,  n_folds=3))
Xtrain.shape, Ttrain.shape, Xval.shape, Tval.shape, Xtest.shape, Ttest.shape

np.unique(Ttrain,  return_counts=True)

 
values, counts = np.unique(Ttrain,  return_counts=True)
counts / Ttrain.shape[0]
values, counts = np.unique(Tval,  return_counts=True)
counts / Tval.shape[0]
n_classes = len(np.unique(T))
nnet = nn.NeuralNetworkClassifier(X.shape[1], [10],  n_classes) 
nnet.train(Xtrain, Ttrain, n_epochs=1000, learning_rate=0.1, method='adam')

plt.figure()
plt.plot(nnet.get_error_trace())
plt.xlabel('Iteration')
plt.ylabel('Data Likelihood');

Classes, Probs = nnet.use(Xtrain)
table = []
for true_class in range(1, 11):
    row = []
    for predicted_class in range(1, 11):
        row.append(f'{100 * np.mean(Classes[Ttrain == true_class] == predicted_class):0.1f}')
    table.append(row)
table

class_names = ('1-Rest', '2-Coloring', '3-Legos', '4-Wii Tennis', '5-Wii Boxing', '6-0.75m/s',
               '7-1.25m/s', '8-1.75m/s', '9-2.25m/s', '10-Stairs')
class_names

import pandas
conf = pandas.DataFrame(table, index=class_names, columns=class_names)
conf
nnet = nn.NeuralNetworkClassifier(X.shape[1], [100, 50],  n_classes) 
nnet.train(Xtrain, Ttrain, n_epochs=100, learning_rate=0.1, method='scg')

plt.figure(figsize=(3, 3))
plt.plot(nnet.get_error_trace())
plt.xlabel('Iteration')
plt.ylabel('Data Likelihood');

print('Training Data')

Classes, Probs = nnet.use(Xtrain)
table = []
for true_class in range(1, 11):
    row = []
    for predicted_class in range(1, 11):
        row.append(f'{100 * np.mean(Classes[Ttrain == true_class] == predicted_class):0.1f}')
    table.append(row)
conf = pandas.DataFrame(table, index=class_names, columns=class_names)
conf

print('Validation Data')

Classes, Probs = nnet.use(Xval)
table = []
for true_class in range(1, 11):
    row = []
    for predicted_class in range(1, 11):
        row.append(f'{100 * np.mean(Classes[Tval == true_class] == predicted_class):0.1f}')
    table.append(row)
conf = pandas.DataFrame(table, index=class_names, columns=class_names)
conf
print('Testing Data')

Classes, Probs = nnet.use(Xtest)
table = []
for true_class in range(1, 11):
    row = []
    for predicted_class in range(1, 11):
        row.append(f'{100 * np.mean(Classes[Ttest == true_class] == predicted_class):0.1f}')
    table.append(row)
conf = pandas.DataFrame(table, index=class_names, columns=class_names)
conf

import scipy.signal as sig

def cwt(eeg, Fs, freqs, width, channelNames=None, graphics=False):
    if freqs.min() == 0:
        print('cwt: Frequencies must be greater than 0.')
        return None, None
    nChannels, nSamples = eeg.shape
    if not channelNames and graphics:
        channelNames = ['Channel {:2d}'.format(i) for i in range(nChannels)]

    nFreqs = len(freqs)
    tfrep = np.zeros((nChannels,  nFreqs, nSamples))
    tfrepPhase = np.zeros((nChannels, nFreqs, nSamples))

    for ch in range(nChannels):
        print('channel', ch, ' freq ', end='')
        for freqi in range(nFreqs):
            print(freqs[freqi], ' ', end='')
            mag, phase = energyvec(freqs[freqi], eeg[ch, :], Fs, width)
            tfrepPhase[ch, freqi, :] = phase
            tfrep[ch, freqi, :] = mag
        print()

    return tfrep, tfrepPhase

def morletLength(Fs, f, width):
  ''' len = morletLength(Fs, f, width) '''
  dt = 1.0/Fs
  sf = f/width
  st = 1.0/(2*np.pi*sf)
  return int((3.5*st - -3.5*st)/dt)

def energyvec(f, s, Fs, width):
  '''
  function [y, phase] <- energyvec(f, s, Fs, width)
  function y <- energyvec(f, s, Fs, width)

  Return a vector containing the energy as a
  function of time for frequency f. The energy
  is calculated using Morlet''s wavelets.
  s : signal
  Fs: sampling frequency
  width : width of Morlet wavelet (><- 5 suggested).
  '''

  dt = 1.0/Fs
  sf = f/float(width)
  st = 1.0/(2*np.pi*sf)

  t = np.arange(-3.5*st, 3.5*st, step=dt)
  m = morlet(f, t, width)
  # yconv = np.convolve(s, m, mode="same")
  yconv = sig.fftconvolve(s, m, mode='same')

  lengthMorlet = len(m)
  firsthalf = int(lengthMorlet/2.0 + 0.5)
  secondhalf = lengthMorlet - firsthalf

  padtotal = len(s) - len(yconv)
  padfront = int(padtotal/2.0 + 0.5)
  padback = padtotal - padfront
  yconvNoBoundary = yconv
  y = np.abs(yconvNoBoundary)**2
  phase = np.angle(yconvNoBoundary, deg=True)
  return y, phase

######################################################################
      
def morlet(f, t, width):
    '''
    function y <- morlet(f, t, width)
    Morlet''s wavelet for frequency f and time t.
    The wavelet will be normalized so the total energy is 1.
    width defines the width of the wavelet.
    A value ><- 5 is suggested.

    Ref: Tallon-Baudry et al., J. Neurosci. 15, 722-734 (1997), page 724

    Ole Jensen, August 1998
    '''
    sf = f/float(width)
    st = 1.0/(2*np.pi*sf)
    A = 1.0/np.sqrt(st*np.sqrt(2*np.pi))
    y = A*np.exp(-t**2/(2*st**2)) * np.exp(1j*2*np.pi*f*t)
    return y

import time
width = 75 
maxFreq = 20
freqs = np.arange(0.5, maxFreq, 0.5) # makes same freqs used in stft above
start = time.time()
tfrep, tfrepPhase = cwt(data[:, 1:].T,  75,  freqs,  width)
print('CWT time: {} seconds'.format(time.time() - start))

plt.figure(figsize=(10, 10))
plt.subplot(5, 1, 1)
plt.plot(data[:, 1:])
plt.axis('tight')

plt.subplot(5, 1, 2)
plt.plot(data[:, 0])
plt.text(5000, 8, '1-Rest, 2-Coloring, 3-Legos, 4-Wii Tennis, 5-Boxing, 6-0.75, 7-1.25 m/s, 8-1.75, 9-2.25 m/s, 10-stairs')
plt.axis('tight')

nSensors = data.shape[1] - 1
for i in range(nSensors):
    plt.subplot(5, 1, i+3)
    plt.imshow(np.log(tfrep[i, :, :]), 
               interpolation='nearest', origin='lower',
               cmap=plt.cm.jet) #plt.cm.Reds)
    plt.xlabel('Seconds')
    plt.ylabel('Frequency in ' + ('$x$', '$y$', '$z$')[i])
    tickstep = round(len(freqs) / 5)
    plt.yticks(np.arange(len(freqs))[::tickstep],
                   [str(i) for i in freqs[::tickstep]])
    plt.axis('auto')
    plt.axis('tight')

X = tfrep.reshape((3*39, -1)).T
X.shape, T.shape, len(np.unique(T))

Xtrain, Ttrain, Xval, Tval, Xtest, Ttest = next(generate_k_fold_cross_validation_sets(X, T, 3))

print(Xtrain.shape)
nnet = nn.NeuralNetworkClassifier(X.shape[1], [10], 10)  #10 classes 
nnet.train(Xtrain, Ttrain, n_epochs = 500, learning_rate=0.1, method='adam')

print('Test Data')

Classes, Probs = nnet.use(Xtest)
table = []
for true_class in range(1, 11):
    row = []
    for predicted_class in range(1, 11):
        row.append(f'{100 * np.mean(Classes[Ttest == true_class] == predicted_class):0.1f}')
    table.append(row)
conf = pandas.DataFrame(table, index=class_names, columns=class_names)
conf
np.mean(Classes == Ttest) * 100




































