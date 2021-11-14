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
 
import numpy as np
import torch

print(torch.__version__)
import time

import matplotlib.pyplot as plt
data = [[0.1, 0.2, 0.3], [1.1, 1.2, 1.3], [2.1, 2.2, 2.3]]
a = np.array(data)
type(a), a.dtype
b = torch.tensor(data)
type(b), b.dtype
c = torch.from_numpy(a)
type(c), c.dtype
d = torch.as_tensor(a)
type(d), d.dtype
e = d.numpy()
type(e), e.dtype
b = torch.from_numpy(a)
a[0,0] = 42.42
b = torch.tensor(a)
a[0,0] = 12345.0

a = np.random.uniform(-0.1, 0.1, size=(10, 5))
b = np.random.uniform(-0.1, 0.1, size=(5, 20))
c = a @ b
c.shape

plt.figure(figsize=(10, 5))
x = np.linspace(-2*np.pi, 2*np.pi, 100)
y = np.sin(x)
dy = np.cos(x)
plt.plot(x, y)
plt.plot(x, dy)
plt.legend(('$\sin(x)$', '$\\frac{d \sin(x)}{dx} = \cos(x)$',));

xt = torch.from_numpy(x)
xt.requires_grad
xt.requires_grad_(True)
xt.requires_grad
yt = torch.sin(xt)

yt.backward(torch.ones(100))
plt.figure(figsize=(10, 5))
plt.plot(xt.detach(), yt.detach())
plt.plot(xt.detach(), xt.grad)
plt.legend(('$\sin(x)$', '$\\frac{d \sin(x)}{dx} = \cos(x)$'));

yt = torch.sin(xt)
yt.backward(torch.ones(100))
xt.grad

plt.figure(figsize=(10, 5))
plt.plot(xt.detach(), yt.detach())
plt.plot(xt.detach(), xt.grad)
plt.legend(('$\sin(x)$', '$\\frac{d \sin(x)}{dx} = \cos(x)$'));

xt.grad.zero_()
for i in range(10):
	yt = torch.sin(xt)
	yt.backward(torch.ones(100))
	print(xt.grad[0])

for i in range(10):
	xt.grad.zero_()
	yt = torch.sin(xt)
	yt.backward(torch.ones(100))
	print(xt.grad[0])

X = np.arange(10).reshape((-1, 1))
T = X ** 2

n_samples = X.shape[0]
n_outputs = T.shape[1]

learning_rate = 0.01 / (n_samples * n_outputs)

W = np.zeros((2, 1))

for epoch in range(100):
	
	Y = X @ W[1:, :] + W[0:1, :]
	
	mse = ((T - Y) ** 2).mean()  # not used
	
	gradient = - X.T @ (T - Y)
	W -= learning_rate * gradient

plt.plot(X, T, 'o-', label='T')
plt.plot(X, Y, 'o-', label='Y')
plt.legend();


X = np.arange(10).reshape((-1, 1))
T = X ** 2
n_samples = X.shape[0]
n_outputs = T.shape[1]

learning_rate = 0.01 / (n_samples * n_outputs)

X = torch.from_numpy(X)
T = torch.from_numpy(T)

W = torch.zeros((2, 1))

for epoch in range(100):
	
	Y = X @ W[1:, :] + W[0:1, :]
	
	mse = ((T - Y) ** 2).mean()  # not used
	
	gradient = - X.T @ (T - Y)
	W -= learning_rate * gradient

plt.plot(X, T, 'o-', label='T')
plt.plot(X, Y, 'o-', label='Y')
plt.legend();

X.dtype

X = np.arange(10).reshape((-1, 1))
T = X ** 2
n_samples = X.shape[0]
n_outputs = T.shape[1]

learning_rate = 0.01 / (n_samples * n_outputs)

X = torch.from_numpy(X).float()  ## ADDED .float()
T = torch.from_numpy(T)

W = torch.zeros((2, 1))

for epoch in range(100):
	
	Y = X @ W[1:, :] + W[0:1, :]
	
	mse = ((T - Y) ** 2).mean()  # not used
		
	gradient = - X.T @ (T - Y)
	W -= learning_rate * gradient

plt.plot(X.detach(), T, 'o-', label='T')
plt.plot(X.detach(), Y.detach(), 'o-', label='Y')
plt.legend();

X = np.arange(10).reshape((-1, 1))
T = X ** 2
n_samples = X.shape[0]
n_outputs = T.shape[1]

learning_rate = 0.01 / (n_samples * n_outputs)

X = torch.from_numpy(X).float()
T = torch.from_numpy(T)

W = torch.zeros((2, 1), requires_grad=True)

for epoch in range(100):

	Y = X @ W[1:, :] + W[0:1, :]
	
	mse = ((T - Y)**2).mean()
	
	mse.backward()  ##  NEW
	
	with torch.no_grad():  ## NEW
		W -= learning_rate * W.grad
		W.grad.zero_()

plt.plot(X.detach(), T, 'o-', label='T')
plt.plot(X.detach(), Y.detach(), 'o-', label='Y')
plt.legend();

X = np.arange(10).reshape((-1, 1))
T = X ** 2
n_samples = X.shape[0]
n_outputs = T.shape[1]

learning_rate = 0.01 / (n_samples * n_outputs)

X = torch.from_numpy(X).float()
T = torch.from_numpy(T)

W = torch.zeros((2, 1), requires_grad=True)

optimizer = torch.optim.SGD([W], lr=learning_rate)   ## NEW

for epoch in range(100):

	Y = X @ W[1:, :] + W[0:1, :]
	
	mse = ((T - Y)**2).mean()
	mse.backward()
	
	optimizer.step()        ## NEW
	optimizer.zero_grad()   ## NEW
	
plt.plot(X.detach(), T, 'o-', label='T')
plt.plot(X.detach(), Y.detach(), 'o-', label='Y')
plt.legend();

# =============================================================================
# X = np.arange(10).reshape((-1, 1))
# T = X ** 2
# n_samples = X.shape[0]
# n_outputs = T.shape[1]
# 
# learning_rate = 0.01 / (n_samples * n_outputs)
# 
# X = torch.from_numpy(X).float()
# T = torch.from_numpy(T)
# 
# W = torch.zeros((2, 1), requires_grad=True)
# 
# optimizer = torch.optim.SGD([W], lr=learning_rate)
# 
# mse_func = torch.nn.MSELoss()  ## NEW
# 
# for epoch in range(100):
# 
#     Y = X @ W[1:, :] + W[0:1, :]
#     
#     mse = mse_func(T, Y)    ## NEW
#     mse.backward()
#     
#     optimizer.step() 
#     optimizer.zero_grad()
#     
# plt.plot(X.detach(), T, 'o-', label='T')
# plt.plot(X.detach(), Y.detach(), 'o-', label='Y')
# plt.legend();
# =============================================================================

X = np.arange(10).reshape((-1, 1))
T = X ** 2
n_samples = X.shape[0]
n_outputs = T.shape[1]

learning_rate = 0.01 / (n_samples * n_outputs)

X = torch.from_numpy(X).float()
T = torch.from_numpy(T).float()   # JUST ADDED THIS

W = torch.zeros((2, 1), requires_grad=True)

optimizer = torch.optim.SGD([W], lr=learning_rate)

mse_func = torch.nn.MSELoss()  ## NEW

for epoch in range(100):

	Y = X @ W[1:, :] + W[0:1, :]
	
	mse = mse_func(T, Y)    ## NEW
	mse.backward()
	
	optimizer.step() 
	optimizer.zero_grad()
	
plt.plot(X.detach(), T, 'o-', label='T')
plt.plot(X.detach(), Y.detach(), 'o-', label='Y')
plt.legend();

n_inputs = 1
n_outputs = 1

model = torch.nn.Sequential(torch.nn.Linear(n_inputs, n_outputs))
model

list(model.parameters())


X = np.arange(10).reshape((-1, 1))
T = X ** 2
n_samples, n_inputs = X.shape  ## NEW, added n_inputs
n_outputs = T.shape[1]

learning_rate = 0.01 / (n_samples * n_outputs)

X = torch.from_numpy(X).float()
T = torch.from_numpy(T).float()

model = torch.nn.Sequential(torch.nn.Linear(n_inputs, n_outputs))

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
mse_func = torch.nn.MSELoss()

for epoch in range(100):

	Y = model(X)            ## NEW
	
	mse = mse_func(T, Y)
	mse.backward()
	
	optimizer.step() 
	optimizer.zero_grad()
	
plt.plot(X.detach(), T, 'o-', label='T')
plt.plot(X.detach(), Y.detach(), 'o-', label='Y')
plt.legend();

X = np.arange(10).reshape((-1, 1))
T = X ** 2
n_samples, n_inputs = X.shape  ## NEW, added n_inputs
n_outputs = T.shape[1]
n_hiddens = [10, 10]

learning_rate = 0.01 / (n_samples * n_outputs)

X = torch.from_numpy(X).float()
T = torch.from_numpy(T).float()

model = torch.nn.Sequential(
	torch.nn.Linear(n_inputs, n_hiddens[0]),
	torch.nn.Tanh(),
	torch.nn.Linear(n_hiddens[0], n_hiddens[1]),
	torch.nn.Tanh(),
	torch.nn.Linear(n_hiddens[1], n_outputs))

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
mse_func = torch.nn.MSELoss()

for epoch in range(1000):

	Y = model(X)            ## NEW
	
	mse = mse_func(T, Y)
	mse.backward()
	
	optimizer.step() 
	optimizer.zero_grad()
	
plt.plot(X.detach(), T, 'o-', label='T')
plt.plot(X.detach(), Y.detach(), 'o-', label='Y')
plt.legend();

X = np.arange(10).reshape((-1, 1))
T = X ** 2
n_samples, n_inputs = X.shape
n_outputs = T.shape[1]
n_hiddens = [10, 10]

learning_rate = 0.5 / (n_samples * n_outputs)  ## Larger learning rate

X = torch.from_numpy(X).float()
T = torch.from_numpy(T).float()

model = torch.nn.Sequential(
	torch.nn.Linear(n_inputs, n_hiddens[0]),
	torch.nn.Tanh(),
	torch.nn.Linear(n_hiddens[0], n_hiddens[1]),
	torch.nn.Tanh(),
	torch.nn.Linear(n_hiddens[1], n_outputs))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
mse_func = torch.nn.MSELoss()

for epoch in range(1000):

	Y = model(X)
	
	mse = mse_func(T, Y)
	mse.backward()
	
	optimizer.step() 
	optimizer.zero_grad()
	
plt.plot(X.detach(), T, 'o-', label='T')
plt.plot(X.detach(), Y.detach(), 'o-', label='Y')
plt.legend();

It is trivial to move data and operations down to a GPU with pytorch.

import time

n = 1000
a = np.random.uniform(-0.1, 0.1, size=(n, n)).astype(np.float32)
b = np.random.uniform(-0.1, 0.1, size=(n, n)).astype(np.float32)

start_time = time.time()
for i in range(1000):
	c = a @ b
elapsed_time = time.time() - start_time

print(f'Took {elapsed_time} seconds')
c.shape

n = 1000
at = (torch.rand(size=(n, n)) - 0.5) * 0.2
bt = (torch.rand(size=(n, n)) - 0.5) * 0.2

# ct = torch.zeros((n, n))

start_time = time.time()
at = at.to('cuda')  ## Don't forget these assignments.  at.to('cuda') does not change at
bt = bt.to('cuda')

start_time = time.time()

for i in range(10000):
	ct = at @ bt

ct = ct.to('cpu')
elapsed_time = time.time() - start_time

print(f'Took {elapsed_time} seconds')
ct.shape

import subprocess

def use_gpu(use=True):
	if use:
		subprocess.run(['system76-power', 'graphics', 'power', 'on'])
		subprocess.run(['sudo', 'modprobe', 'nvidia'])
	else:
		subprocess.run(['sudo', 'rmmod', 'nvidia'])
		subprocess.run(['system76-power', 'graphics', 'off'])
		
# use_gpu()  #  if running on my system76 laptop

torch.cuda.is_available()

use_gpu = False

n_samples = 10000
X = np.linspace(0, 10, n_samples).reshape((-1, 1))
T = X ** 2
n_samples, n_inputs = X.shape 
n_outputs = T.shape[1]

n_hiddens = [100, 100]

learning_rate = 0.1 #  / (n_samples * n_outputs)  ## Larger learning rate

X = torch.from_numpy(X).float()
T = torch.from_numpy(T).float()

model = torch.nn.Sequential(
	torch.nn.Linear(n_inputs, n_hiddens[0]),
	torch.nn.Tanh(),
	torch.nn.Linear(n_hiddens[0], n_hiddens[1]),
	torch.nn.Tanh(),
	torch.nn.Linear(n_hiddens[1], n_outputs))

if use_gpu:
	print('Moving data and model to GPU')
	X = X.to('cuda')
	T = T.to('cuda')
	model.to('cuda')

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
mse_func = torch.nn.MSELoss()

start_time = time.time()

for epoch in range(1000):

	Y = model(X)
	
	mse = mse_func(T, Y)
	mse.backward()
	
	optimizer.step() 
	optimizer.zero_grad()

elapsed_time = time.time() - start_time
print(f'Training took {elapsed_time:.2f} seconds.')
if use_gpu:
	print('   with GPU')
	
plt.plot(X.cpu().detach(), T.cpu(), 'o-', label='T')
plt.plot(X.cpu().detach(), Y.cpu().detach(), 'o-', label='Y')
plt.legend();



use_gpu = True

n_samples = 10000
X = np.linspace(0, 10, n_samples).reshape((-1, 1))
T = X ** 2
n_samples, n_inputs = X.shape 
n_outputs = T.shape[1]

n_hiddens = [100, 100]

learning_rate = 0.1 #  / (n_samples * n_outputs)  ## Larger learning rate

X = torch.from_numpy(X).float()
T = torch.from_numpy(T).float()

model = torch.nn.Sequential(
	torch.nn.Linear(n_inputs, n_hiddens[0]),
	torch.nn.Tanh(),
	torch.nn.Linear(n_hiddens[0], n_hiddens[1]),
	torch.nn.Tanh(),
	torch.nn.Linear(n_hiddens[1], n_outputs))

if use_gpu:
	print('Moving data and model to GPU')
	X = X.to('cuda')
	T = T.to('cuda')
	model.to('cuda')

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
mse_func = torch.nn.MSELoss()

start_time = time.time()

for epoch in range(1000):

	Y = model(X) 
	
	mse = mse_func(T, Y)
	mse.backward()
	
	optimizer.step() 
	optimizer.zero_grad()

elapsed_time = time.time() - start_time
print(f'Training took {elapsed_time:.2f} seconds.')
if use_gpu:
	print('   with GPU')
	
plt.plot(X.cpu().detach(), T.cpu(), 'o-', label='T')
plt.plot(X.cpu().detach(), Y.cpu().detach(), 'o-', label='Y')
plt.legend();

use_gpu = True

n_samples = 10000
X = np.linspace(0, 10, n_samples).reshape((-1, 1))
T = X ** 2
n_samples, n_inputs = X.shape 
n_outputs = T.shape[1]


X = torch.from_numpy(X).float()
T = torch.from_numpy(T).float()

class NNet(torch.nn.Module):
	
	def __init__(self, n_inputs, n_hiddens_list, n_outputs):
		super().__init__()  # call parent class (torch.nn.Module) constructor
			
		self.hidden_layers = torch.nn.ModuleList()  # necessary for model.to('cuda')
		for nh in n_hiddens_list:
			self.hidden_layers.append( torch.nn.Sequential(
				torch.nn.Linear(n_inputs, nh),
				torch.nn.Tanh()))
			
			n_inputs = nh
		self.output_layer = torch.nn.Linear(n_inputs, n_outputs)
			
	def forward(self, X):
		Y = X
		for hidden_layer in self.hidden_layers:
			Y = hidden_layer(Y)
		Y = self.output_layer(Y)
		return Y

n_hiddens = [100, 100]

learning_rate = 0.1 #  / (n_samples * n_outputs)  ## Larger learning rate

model = NNet(n_inputs, n_hiddens, n_outputs)

if use_gpu:
	print('Moving data and model to GPU')
	X = X.to('cuda')
	T = T.to('cuda')
	model.to('cuda')   # or   model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
mse_func = torch.nn.MSELoss()

start_time = time.time()

for epoch in range(1000):

	Y = model(X) 
	
	mse = mse_func(T, Y)
	mse.backward()
	
	optimizer.step() 
	optimizer.zero_grad()

elapsed_time = time.time() - start_time
print(f'Training took {elapsed_time:.2f} seconds.')
if use_gpu:
	print('   with GPU')
	
plt.plot(X.cpu().detach(), T.cpu(), 'o-', label='T')
plt.plot(X.cpu().detach(), Y.cpu().detach(), 'o-', label='Y')
plt.legend();



 