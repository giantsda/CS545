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





class Optimizers():

    def __init__(self, all_weights):
        '''all_weights is a vector of all of a neural networks weights concatenated into a one-dimensional vector'''
        
        self.all_weights = all_weights

        self.sgd_initialized = False
        self.scg_initialized = False
        self.adam_initialized = False
 
def parabola(xmin, s):
    d = x.reshape(-1, 1) - xmin
    return d.T @ s @ d

def parabola_gradient(xmin, s):
    d = x.reshape(-1, 1) - xmin
    return 2 * (s @ d).reshape(-1)  # must be row vector


def rosen():
    v = (1.0 - x[0])**2 + 100 * ((x[1] - x[0]**2)**2)
    return v

def rosen_gradient():
    g1 = -400 * (x[1] - x[0]**2) * x[0] - 2 * (1 - x[0])
    g2 =  200 * (x[1] - x[0]**2)
    return np.array([g1, g2])

def show_trace(fig, x_start, function, function_gradient, function_args, n_epochs, learning_rates):
    global x
    
    x = x_start.copy()
    sgd_weights_trace = [x.copy()]
    def callback(epoch):
        sgd_weights_trace.append(x.copy())
    optimizer = opt.Optimizers(x)
    errors_sgd = optimizer.sgd(function, function_gradient, function_args,
                               n_epochs=n_epochs, learning_rate=learning_rates[0],
                               callback_f=callback, verbose=False)
    
    x = x_start.copy()
    adam_weights_trace = [x.copy()]
    def callback(epoch):
        adam_weights_trace.append(x.copy())
    optimizer = opt.Optimizers(x)
    errors_adam = optimizer.adam(function, function_gradient, function_args,
                                 n_epochs=n_epochs, learning_rate=learning_rates[1],
                                 callback_f=callback, verbose=False)
    
    x = x_start.copy()
    scg_weights_trace = [x.copy()]
    def callback(epoch):
        scg_weights_trace.append(x.copy())
    optimizer = opt.Optimizers(x)
    errors_scg = optimizer.scg(function, function_gradient, function_args,
                               n_epochs=200,
                               callback_f=callback, verbose=False)

    plt.clf()
    
    xt = np.array(sgd_weights_trace)
    plt.plot(xt[:, 0], xt[:, 1], 'ro-', alpha=0.4, label='SGD')

    xt = np.array(adam_weights_trace)
    plt.plot(xt[:, 0], xt[:, 1], 'go-', alpha=0.2, label='Adam')

    xt = np.array(scg_weights_trace)
    plt.plot(xt[:, 0], xt[:, 1], 'ko-', alpha=0.4, label='SCG')

    plt.contourf(X, Y, Z, 20, alpha=0.3)
    plt.axis('tight')
    
    plt.legend()
    ipd.clear_output(wait=True)
    ipd.display(fig) 

# %% main functions


center = np.array([5, 5]).reshape(2, 1)
S = np.array([[5, 3], [3, 5]])

n = 20
xs = np.linspace(0, 10, n)
ys = np.linspace(0, 10, n)
X,Y = np.meshgrid(xs, ys)
both = np.vstack((X.flat, Y.flat)).T
nall = n * n

Z = np.zeros(nall)
for i in range(nall):
    x = both[i:i + 1, :]
    Z[i] = parabola(center, S)
Z = Z.reshape(n, n)

# see https://matplotlib.org/3.1.0/gallery/mplot3d/surface3d.html
fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection='3d')

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False,
                       cmap=plt.cm.coolwarm)
ax.view_init(elev=30., azim=100)



fig = plt.figure(figsize=(10, 10))

x_start = np.random.uniform(0, 10, 2)
show_trace(fig, x_start, parabola, parabola_gradient, [center, S], 
           n_epochs=200, learning_rates=[0.01, 0.5])

ipd.clear_output(wait=True) 


 

n = 10
xmin, xmax = -1,2
xs = np.linspace(xmin, xmax, n)
ys = np.linspace(xmin, xmax, n)
X, Y = np.meshgrid(xs, ys)    
    
both = np.vstack((X.flat, Y.flat)).T
nall = n * n
Z = np.zeros(nall)
for i in range(n * n):
    x = both[i]
    Z[i] = rosen()
Z.resize((n, n))

fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection='3d')

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False,
                       cmap=plt.cm.coolwarm)
ax.view_init(elev=40., azim=260)



fig = plt.figure(figsize=(10, 10))

x_start = np.random.uniform(-1, 2, 2)
show_trace(fig, x_start, rosen, rosen_gradient, [], n_epochs=400, 
           learning_rates=[0.0001, 0.5])
    
ipd.clear_output(wait=True) 





adfasd=1