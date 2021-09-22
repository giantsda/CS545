#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 09:55:02 2021

@author: chen
"""
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
import pandas  # for reading csv file



class NeuralNetwork():
	"""
	A class that represents a neural network for nonlinear regression

	Attributes
	----------
	n_inputs : int
		The number of values in each sample
	n_hidden_units_by_layers: list of ints, or empty
		The number of units in each hidden layer.
		Its length specifies the number of hidden layers.
	n_outputs: int
		The number of units in output layer
	all_weights : one-dimensional numpy array
		Contains all weights of the network as a vector
	Ws : list of two-dimensional numpy arrays
		Contains matrices of weights in each layer,
		as views into all_weights
	all_gradients : one-dimensional numpy array
		Contains all gradients of mean square error with
		respect to each weight in the network as a vector
	Grads : list of two-dimensional numpy arrays
		Contains matrices of gradients weights in each layer,
		as views into all_gradients
	total_epochs : int
		Total number of epochs trained so far
	error_trace : list
		Mean square error (standardized) after each epoch
	X_means : one-dimensional numpy array
		Means of the components, or features, across samples
	X_stds : one-dimensional numpy array
		Standard deviations of the components, or features, across samples
	T_means : one-dimensional numpy array
		Means of the components of the targets, across samples
	T_stds : one-dimensional numpy array
		Standard deviations of the components of the targets, across samples
		
	Methods
	-------
	make_weights_and_views(shapes)
		Creates all initial weights and views for each layer

	train(X, T, n_epochs, method='sgd', learning_rate=None, verbose=True)
		Trains the network using samples by rows in X and T

	use(X)
		Applies network to inputs X and returns network's output
	""" 
	def __init__(self, n_inputs, n_hidden_units_by_layers, n_outputs):

		
		self.n_inputs = n_inputs
		self.n_outputs = n_outputs
		self.n_hidden_units_by_layers = n_hidden_units_by_layers
		self.Ws= []
		self.all_weights=np.empty([0,1])
		self.shapes=[]
 
	
		layer_n=n_inputs
		for layerI in range(len(self.n_hidden_units_by_layers)):
			layerI_N=self.n_hidden_units_by_layers[layerI]
			self.shapes.append([1 + layer_n, layerI_N])
			layer_n=layerI_N
		self.shapes.append([1 + layer_n, n_outputs])
		
		self.make_weights_and_views(self.shapes)
		self.all_gradients = []
		self.Grads = []
		self.total_epochs = 0
		self.error_trace = []
		self.X_means = None
		self.X_stds = None
		self.T_means = None
		self.T_stds = None
		self.Ys=None

	def make_weights_and_views(self, shapes):
		"""Creates vector of all weights and views for each layer

		Parameters
		----------
		shapes : list of pairs of ints
			Each pair is number of rows and columns of weights in each layer

		Returns
		-------
		Vector of all weights, and list of views into this vector for each layer
		"""

		# Create one-dimensional numpy array of all weights with random initial values
		#  ...

		# Build list of views by reshaping corresponding elements
		# from vector of all weights into correct shape for each layer.        
		# ...
 
		for layerI in range(len(shapes)):
			shapeX=shapes[layerI][0]
			shapeY=shapes[layerI][1]
			wI=1 / np.sqrt(shapeX) * np.random.uniform(-1, 1, size=(shapeX, shapeY))
			self.Ws.append(wI)
# 			haha=np.vstack((self.all_weights,wI.reshape(-1,1)))
			self.all_weights=np.vstack((self.all_weights,wI.reshape(-1,1))) 
		self.all_weights=self.all_weights.flatten()
			
			
			
	def __repr__(self):
		return f'NeuralNetwork({self.n_inputs}, ' + \
			f'{self.n_hidden_units_by_layers}, {self.n_outputs})'

	def __str__(self):
		s = self.__repr__()
		if self.total_epochs > 0:
			s += f'\n Trained for {self.total_epochs} epochs.'
			s += f'\n Final standardized training error {self.error_trace[-1]:.4g}.'
		return s
 
	def train(self, X, T, n_epochs, method='sgd', learning_rate=None, verbose=True):
		"""Updates the weights 
		Parameters
		----------
		X : two-dimensional numpy array
			number of samples  x  number of input components
		T : two-dimensional numpy array
			number of samples  x  number of output components
		n_epochs : int
			Number of passes to take through all samples
		method : str
			'sgd', 'adam', or 'scg'
		learning_rate : float
			Controls the step size of each update, only for sgd and adam
		verbose: boolean
			If True, progress is shown with print statements
		"""

		# Calculate and assign standardization parameters
		# ...
		
		self.X_means = np.mean(X, axis=0)
		self.X_stds = np.std(X, axis=0)
		self.T_means = np.mean(T, axis=0)
		self.T_stds = np.std(T, axis=0)
		# Standardize X and T

		X = (X - self.X_means) / self.X_stds
		T = (T - self.T_means) / self.T_stds
 
		# Instantiate Optimizers object by giving it vector of all weights
		optimizer = opt.Optimizers(self.all_weights)

		error_convert_f = lambda err: (np.sqrt(err) * self.T_stds)[0]
		
		# Call the requested optimizer method to train the weights.

		if method == 'sgd':
			error_trace=optimizer.sgd(self.error_f, self.gradient_f,fargs=[X,T],error_convert_f=error_convert_f,learning_rate=learning_rate,n_epochs=n_epochs,verbose=True)
		elif method == 'adam':
			error_trace=optimizer.adam(self.error_f, self.gradient_f,fargs=[X,T],error_convert_f=error_convert_f,learning_rate=learning_rate,n_epochs=n_epochs)
		elif method == 'scg':
			error_trace=optimizer.scg(self.error_f, self.gradient_f,fargs=[X,T],n_epochs=n_epochs)
		else:
			raise Exception("method must be 'sgd', 'adam', or 'scg'")
 
		self.total_epochs += len(error_trace)
		self.error_trace = error_trace



		self._forward(X)
		error = (T - self.Ys[-1]) * self.T_stds 
# 		plt.plot(X, self.Ys[-1], 'o-', label='Model ')
# 		errors.append(nnet.get_error_trace())
# 		plt.plot(X, T, '*-', label='Train')
# 		print(self.all_weights[20])
# 		plt.plot(self.all_weights, '*-', label='w')

# 		plt.show()

# 		plt.draw()
# 		plt.pause(0.00000001)
# 		plt.clf()
 
		
		
		
		
		# Return neural network object to allow applying other methods
		# after training, such as:    Y = nnet.train(X, T, 100, 0.01).use(X)

		return self

	def addOnes(self,A):
		return np.insert(A, 0, 1, axis=1)
	
	def _forward(self, X):
		"""Calculate outputs of each layer given inputs in X
		
		Parameters
		----------
		X : input samples, standardized

		Returns
		-------
		Outputs of all layers as list
		"""
		# unpack self.all_weights to self.Ws
		i=0
		for layerI in range(len(self.shapes)):
			shapeX=self.shapes[layerI][0]
			shapeY=self.shapes[layerI][1]
			self.Ws[layerI]=self.all_weights[i:i+shapeX*shapeY].reshape(shapeX,shapeY)
			i+=shapeX*shapeY
		
		self.Ys=[]
		for layerI in range(len(self.n_hidden_units_by_layers)):
			X=np.tanh(self.addOnes(X) @ self.Ws[layerI])
			self.Ys.append(X)
		X=self.addOnes(X)@self.Ws[-1]
		self.Ys.append(X)
		# Append output of each layer to list in self.Ys, then return it.
		# ...

	# Function to be minimized by optimizer method, mean squared error
	def error_f(self, X, T):
		"""Calculate output of net and its mean squared error 

		Parameters
		----------
		X : two-dimensional numpy array
			number of samples  x  number of input components
		T : two-dimensional numpy array
			number of samples  x  number of output components

		Returns
		-------
		Mean square error as scalar float that is the mean
		square error over all samples
		"""
		self._forward(X)
# 		error = (T - self.Ys[-1]) * self.T_stds  # I cannt see why I * T_stds here.

		error = (T - self.Ys[-1])

		MSE= np.mean(error**2)
		
		self.error_trace.append(MSE)
		
		return MSE
		
		# Call _forward, calculate mean square error and return it.
		# ...

	# Gradient of function to be minimized for use by optimizer method
	def gradient_f(self, X, T):
		"""Returns gradient wrt all weights. Assumes _forward already called.

		Parameters
		----------
		X : two-dimensional numpy array
			number of samples  x  number of input components
		T : two-dimensional numpy array
			number of samples  x  number of output components

		Returns
		-------
		Vector of gradients of mean square error wrt all weights
		"""
 
		self._forward( X)
		# Assumes forward_pass just called with layer outputs saved in self.Ys.
		n_samples = X.shape[0]
		n_outputs = T.shape[1]
		n_layers = len(self.n_hidden_units_by_layers) + 1

		# D is delta matrix to be back propagated
		D = (T - self.Ys[-1]) /(n_samples * n_outputs) 
		self.Grads =  [None] *n_layers

		# Step backwards through the layers to back-propagate the error (D)
		for layeri in range(n_layers - 1, -1, -1):
			# gradient of all but bias weights
 
			# Back-propagate this layer's delta to previous layer
			if layeri > 0:
				self.Grads[layeri]= -self.addOnes(self.Ys[layeri-1]).T@D
				D =D@self.Ws[layeri][1:,:].T*(1-self.Ys[layeri-1]**2)  
			else:
				self.Grads[layeri]= -self.addOnes(X).T@D
		 
		

# 		self.all_gradients=


		self.all_gradients=np.empty([0,1])
		for layerI in range(n_layers):
			 
# 			haha=np.vstack((self.all_weights,wI.reshape(-1,1)))
			self.all_gradients=np.vstack((self.all_gradients,self.Grads[layerI].reshape(-1,1))) 
			
		self.all_gradients=self.all_gradients.flatten();
			
			
			
		return self.all_gradients

	def use(self, X):
		"""Return the output of the network for input samples as rows in X

		Parameters
		----------
		X : two-dimensional numpy array
			number of samples  x  number of input components, unstandardized

		Returns
		-------
		Output of neural network, unstandardized, as numpy array
		of shape  number of samples  x  number of outputs
		"""
 
		X=(X-self.X_means)/self.X_stds
		self._forward( X)
		Y=self.Ys[-1]
		Y=Y*self.T_stds+self.T_means
		return Y 
 
	def get_error_trace(self):
		"""Returns list of standardized mean square error for each epoch"""
		return self.error_trace

 
 
# =============================================================================
# #%% main functions
# X = np.arange(-2, 2, 0.05).reshape(-1, 1)
# T = np.sin(X) * np.sin(X * 10)
# 
# errors = []
# n_epochs = 10000
# method_rhos = [ ('adam',0.01),
#                
#                 ('scg', None)]
# 
# for method, rho in method_rhos:
# 	nnet = NeuralNetwork(X.shape[1], [30, 30], 1)
# 	nnet.train(X, T, n_epochs, method=method, learning_rate=rho)
# 	Y = nnet.use(X)
# 	plt.plot(X, Y, 'o-', label='Model ' + method)
# 	plt.plot(X, T, 'o', label='Train')
# 	errors.append(nnet.get_error_trace())
# 	plt.show()
# 	exit()
# 
# 
# plt.plot(X, T, 'o', label='Train')
# plt.xlabel('X')
# plt.ylabel('T or Y')
# plt.legend();
# 
# 
# 
# exit()
# =============================================================================

#%% Boston House Price
data = pandas.read_csv('boston.csv', delimiter=',', decimal='.', usecols=range(14), na_values=-200)
data = data.dropna(axis=0)
 
data=data.to_numpy()
 
X=data[:,0:13]
T=data[:,-1].reshape(-1,1)

def partition(X, T, train_fraction):
	n_samples = X.shape[0]
	rows = np.arange(n_samples)
	np.random.shuffle(rows)
	
	n_train = round(n_samples * train_fraction)
	
	Xtrain = X[rows[:n_train], :]
	Ttrain = T[rows[:n_train], :]
	Xtest = X[rows[n_train:], :]
	Ttest = T[rows[n_train:], :]
	return [Xtrain, Ttrain, Xtest, Ttest]
def rmse(T, Y):
	return np.sqrt(np.mean((T - Y)**2))

Xtrain, Ttrain, Xtest, Ttest = partition(X, T, 0.8)  

errors = []
layersS=[ [10,10],
	 [5, 5, 5],
	 [20,20],
	 ]

n_epochs = 10000
method_rhos = [  ('sgd',0.01),
				('adam',0.01),
					('scg', None)]
	 
 

for i in range(len(layersS)):
	for j in range (len(method_rhos)):
		method=method_rhos[j][0]
		rho=method_rhos[j][1]
		layer=layersS[i]
		nnet =NeuralNetwork(X.shape[1], layer,1)
		nnet.train(X, T, n_epochs, method=method, learning_rate=rho)
		Y = nnet.use(X)
		plt.plot(nnet.get_error_trace(), label='Model ' + method)
	 
		errors.append(nnet.get_error_trace())
plt.show()
		 


 