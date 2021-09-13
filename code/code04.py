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



def f(s):
	return 1 / (1 + np.exp(-s))

def df(f_value):
	return f_value * (1 - f_value)



X = np.linspace(0, 10, 20).reshape((-1,1))
T = X * 0.1 + np.random.uniform(-0.4, 0.4, size=(20, 1))
plt.plot(X, T, 'o');


w = np.zeros((2, 1))
rho = 0.1
X1 = np.insert(X, 0, 1, 1)
for iter in range(100):
	for n in range(X.shape[0]):
		Xn = X1[n:n + 1, :]
		Tn = T[n:n + 1, :]
		yn = f(Xn @ w)
		w += rho * Xn.T * (Tn - yn) * df(yn) 
	
plt.plot(X, T, 'o')
plt.plot(X, f(X1 @ w))





# =============================================================================
# 2 layer network
# =============================================================================

# Add constant column of 1's
def addOnes(A):
	return np.insert(A, 0, 1, axis=1)

def rmse(T, Y, Tstds):
	error = (T - Y) * Tstds 
	return np.sqrt(np.mean(error ** 2))


# Make some training data
n = 30
Xtrain = np.linspace(0., 20.0, n).reshape((n, 1)) - 10
Ttrain = 0.2 + 0.05 * (Xtrain + 10) + 0.4 * np.sin(Xtrain + 10) + 0.2 * np.random.normal(size=(n, 1))

# Make some testing data
Xtest = Xtrain + 0.1 * np.random.normal(size=(n, 1))
Ttest = 0.2 + 0.05 * (Xtest + 10) + 0.4 * np.sin(Xtest + 10) + 0.2 * np.random.normal(size=(n, 1))

nSamples = Xtrain.shape[0]
nOutputs = Ttrain.shape[1]


Xmeans = Xtrain.mean(axis=0)
Xstds = Xtrain.std(axis=0)
Tmeans = Ttrain.mean(axis=0)
Tstds = Ttrain.std(axis=0)

XtrainS = (Xtrain - Xmeans) / Xstds
TtrainS = (Ttrain - Tmeans) / Tstds
XtestS = (Xtest - Xmeans) / Xstds
TtestS = (Ttest - Tmeans) / Tstds

XtrainS1 = addOnes(XtrainS)
XtestS1 = addOnes(XtestS)

# Set parameters of neural network
n_hiddens = 20

n_samples, n_outputs = Ttrain.shape

rho_h = 0.5
rho_o = 0.1

rho_h = rho_h / (n_samples * n_outputs)
rho_o = rho_o / (n_samples * n_outputs)

# Initialize weights to uniformly distributed values between small normally-distributed between -0.1 and 0.1
V = np.random.uniform(-1, 1, size=(1 + 1, n_hiddens)) / np.sqrt(XtrainS1.shape[1])
W = np.random.uniform(-1, 1, size=(1 + n_hiddens, n_outputs)) / np.sqrt(n_hiddens + 1)

# Take n_epochs steepest descent steps in gradient descent search in mean-squared-error function
n_epochs = 100000

# collect training and testing errors for plotting
error_trace = []

fig = plt.figure(figsize=(10, 20))

for epoch in range(n_epochs):

	# Function we wish to minimize, mean squared error
	# ------------------------------------------------
	# Forward pass on all training data
	Z = np.tanh(XtrainS1 @ V)
	Z1 = addOnes(Z)
	Y = Z1 @ W
	mse = np.mean((TtrainS - Y)**2)
	
	# Gradient of mean squared error with respect to V and W
	# ------------------------------------------------------
	Dw = TtrainS - Y
	Dv = Dw @ W[1:, :].T * (1 - Z**2)
	grad_wrt_W = - Z1.T @ Dw
	grad_wrt_V = - XtrainS1.T @ Dv
	
	# Take step down the gradient
	W = W - rho_o * grad_wrt_W
	V = V - rho_h * grad_wrt_V  

	# Apply model with new weights to train and test data, calculate the RMSEs and append to error_trace
	YtrainS = addOnes(np.tanh(XtrainS1 @ V)) @ W    # Forward pass in one line !!
	YtestS = addOnes(np.tanh(XtestS1 @ V)) @ W 
	error_trace.append([rmse(TtrainS, YtrainS, Tstds),
			rmse(TtestS, YtestS, Tstds)])
	
	if epoch % 2000 == 0 or epoch == n_epochs - 1:
		plt.clf()
		plt.subplot(3, 1, 1)
		plt.plot(np.array(error_trace)[:epoch, :])
		plt.ylim(0, 0.7)
		plt.xlabel('Epochs')
		plt.ylabel('RMSE')
		plt.legend(('Train','Test'), loc='upper left')
		
		plt.subplot(3, 1, 2)
		Ytest = YtestS * Tstds + Tmeans
		plt.plot(Xtrain, Ttrain, 'o-', Xtest, Ttest, 'o-', Xtest, Ytest, 'o-')
		plt.xlim(-10, 10)
		plt.legend(('Training', 'Testing', 'Model'), loc='upper left')
		plt.xlabel('$x$')
		plt.ylabel('Actual and Predicted $f(x)$')
		
		plt.subplot(3, 1, 3)
		plt.plot(Xtrain, Z)
		plt.ylim(-1.1, 1.1)
		plt.xlabel('$x$')
		plt.ylabel('Hidden Unit Outputs ($z$)');
		
		ipd.clear_output(wait=True)
		plt.pause(0.001)
		ipd.display(fig)
						
 
ipd.clear_output(wait=True)








