import numpy as np
import optimizers as opt
import sys  # for sys.float_info.epsilon
 
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
 


######################################################################
## class NeuralNetwork()
######################################################################

class NeuralNetwork():

	def __init__(self, n_inputs, n_hidden_units_by_layers, n_outputs):
		'''
		n_inputs: int
		n_hidden_units_by_layers: list of ints, or empty
		n_outputs: int
		'''

		self.n_inputs = n_inputs
		self.n_hidden_units_by_layers = n_hidden_units_by_layers
		self.n_outputs = n_outputs

		# Build list of shapes for weight matrices in each layera
		shapes = []
		n_in = n_inputs
		for nu in self.n_hidden_units_by_layers + [n_outputs]:
			shapes.append((n_in + 1, nu))
			n_in = nu

		self.all_weights, self.Ws = self._make_weights_and_views(shapes)
		self.all_gradients, self.Grads = self._make_weights_and_views(shapes)

		self.total_epochs = 0
		self.error_trace = []
		self.X_means = None
		self.X_stds = None
		self.T_means = None
		self.T_stds = None

	def _make_weights_and_views(self, shapes):
		'''
		shapes: list of pairs of ints for number of rows and columns
				in each layer
		Returns vector of all weights, and views into this vector
				for each layer
		'''
		all_weights = np.hstack([np.random.uniform(size=shape).flat
								 / np.sqrt(shape[0])
								 for shape in shapes])
		# Build list of views by reshaping corresponding elements
		# from vector of all weights into correct shape for each layer.
		views = []
		first_element = 0
		for shape in shapes:
			n_elements = shape[0] * shape[1]
			last_element = first_element + n_elements
			views.append(all_weights[first_element:last_element]
						 .reshape(shape))
			first_element = last_element

		return all_weights, views

	def __repr__(self):
		return f'NeuralNetwork({self.n_inputs}, ' + \
			f'{self.n_hidden_units_by_layers}, {self.n_outputs})'

	def __str__(self):
		s = self.__repr__()
		if self.total_epochs > 0:
			s += f'\n Trained for {self.total_epochs} epochs.'
			s += f'\n Final standardized training error {self.error_trace[-1]:.4g}.'
		return s

	def addOnes(self,A):
		return np.insert(A, 0, 1, axis=1)
	
	
	def train(self, X, T, n_epochs, method='sgd', learning_rate=None, verbose=True):
		'''
		X: n_samples x n_inputs matrix of input samples, one per row
		T: n_samples x n_outputs matrix of target output values,
			one sample per row
		n_epochs: number of passes to take through all samples
			updating weights each pass
		method: 'sgd', 'adam', or 'scg'
		learning_rate: factor controlling the step size of each update
		'''

		# Setup standardization parameters
		# Setup standardization parameters
		if self.X_means is None:
			self.X_means = X.mean(axis=0)
			self.X_stds = X.std(axis=0)
			self.X_stds[self.X_stds == 0] = 1
			self.T_means = T.mean(axis=0)
			self.T_stds = T.std(axis=0)

		# Standardize X and T
		X = (X - self.X_means) / self.X_stds
		T = (T - self.T_means) / self.T_stds

		# Instantiate Optimizers object by giving it vector of all weights
		optimizer = opt.Optimizers(self.all_weights)

		_error_convert_f = lambda err: (np.sqrt(err) * self.T_stds)[0]

		if method == 'sgd':

			error_trace = optimizer.sgd(self._error_f, self._gradient_f,
										fargs=[X, T], n_epochs=n_epochs,
										learning_rate=learning_rate,
										error_convert_f=_error_convert_f,
										verbose=verbose)

		elif method == 'adam':

			error_trace = optimizer.adam(self._error_f, self._gradient_f,
										 fargs=[X, T], n_epochs=n_epochs,
										 learning_rate=learning_rate,
										 error_convert_f=_error_convert_f,
										 verbose=verbose)

		elif method == 'scg':

			error_trace = optimizer.scg(self._error_f, self._gradient_f,
										fargs=[X, T], n_epochs=n_epochs,
										error_convert_f=_error_convert_f,
										verbose=verbose)

		else:
			raise Exception("method must be 'sgd', 'adam', or 'scg'")

		self.total_epochs += len(error_trace)
		self.error_trace += error_trace

		# Return neural network object to allow applying other methods
		# after training, such as:    Y = nnet.train(X, T, 100, 0.01).use(X)

		return self

	def _forward(self, X): #implictly addOnes inside
		'''
		X assumed to be standardized and with first column of 1's
		'''
		self.Ys = [X]
		for W in self.Ws[:-1]:  # forward through all but last layer
			self.Ys.append(np.tanh(self.Ys[-1] @ W[1:, :] + W[0:1, :])) # addOnes is implemented implictly here. No need to add addOnes
		last_W = self.Ws[-1]
		self.Ys.append(self.Ys[-1] @ last_W[1:, :] + last_W[0:1, :])
		return self.Ys

	# Function to be minimized by optimizer method, mean squared error
	def _error_f(self, X, T):
		Ys = self._forward(X)
		mean_sq_error = np.mean((T - Ys[-1]) ** 2)
		return mean_sq_error

	# Gradient of function to be minimized for use by optimizer method
	def _gradient_f(self, X, T):
		# Assumes forward_pass just called with layer outputs saved in self.Ys.
		n_samples = X.shape[0]
		n_outputs = T.shape[1]

		# D is delta matrix to be back propagated
		D = -(T - self.Ys[-1]) / (n_samples * n_outputs)
		self._backpropagate(D)

		return self.all_gradients

	def _backpropagate(self, D):
		# Step backwards through the layers to back-propagate the error (D)
		n_layers = len(self.n_hidden_units_by_layers) + 1
		for layeri in range(n_layers - 1, -1, -1):
			# gradient of all but bias weights
			self.Grads[layeri][1:, :] = self.Ys[layeri].T @ D
			# gradient of just the bias weights
			self.Grads[layeri][0:1, :] = np.sum(D, axis=0)
			# Back-propagate this layer's delta to previous layer
			if layeri > 0:
				D = D @ self.Ws[layeri][1:, :].T * (1 - self.Ys[layeri] ** 2)

	def use(self, X):
		'''X assumed to not be standardized'''
		# Standardize X
		X = (X - self.X_means) / self.X_stds
		Ys = self._forward(X)
		# Unstandardize output Y before returning it
		return Ys[-1] * self.T_stds + self.T_means

	def get_error_trace(self):
		return self.error_trace

class NeuralNetworkClassifier(NeuralNetwork):

	def makeIndicatorVars(self, T):
		# Make sure T is two-dimensional. Should be nSamples x 1.
		if T.ndim == 1:
			T = T.reshape((-1, 1))    
		return (T == np.unique(T)).astype(int)

	def train(self, X, T, n_epochs, method='sgd', learning_rate=None, verbose=True):
		'''
		X: n_samples x n_inputs matrix of input samples, one per row
		T: n_samples x n_outputs matrix of target output values,
			one sample per row
		n_epochs: number of passes to take through all samples
			updating weights each pass
		method: 'sgd', 'adam', or 'scg'
		learning_rate: factor controlling the step size of each update
		'''

		# Setup standardization parameters
		# Setup standardization parameters
		
		T_save=T.copy()   # save before destorying T
		
		T = self.makeIndicatorVars(T)
		
		
		
		if self.X_means is None:
			self.X_means = X.mean(axis=0)
			self.X_stds = X.std(axis=0)
			self.X_stds[self.X_stds == 0] = 1
# 			self.T_means = T.mean(axis=0)   # classfier does not need to be standardlized, they are 0, 1, or 2, or 3, ...
# 			self.T_stds = T.std(axis=0)

		# Standardize X
		X = (X - self.X_means) / self.X_stds
 
		# Instantiate Optimizers object by giving it vector of all weights
		optimizer = opt.Optimizers(self.all_weights)

		to_likelihood = lambda nll: np.exp(-nll)

# 		error=self._neg_log_likelihood_f(X, T)
# 		print(to_likelihood(error)
		
		if method == 'sgd':

			error_trace = optimizer.sgd(self._neg_log_likelihood_f, self._gradient_f,
										fargs=[X, T], n_epochs=n_epochs,
										learning_rate=learning_rate,
										error_convert_f=to_likelihood,
										verbose=verbose)

		elif method == 'adam':

			error_trace = optimizer.adam(self._neg_log_likelihood_f, self._gradient_f,
										 fargs=[X, T], n_epochs=n_epochs,
										 learning_rate=learning_rate,
										 error_convert_f=to_likelihood,
										 verbose=verbose)

		elif method == 'scg':

			error_trace = optimizer.scg(self._neg_log_likelihood_f, self._gradient_f,
										fargs=[X, T], n_epochs=n_epochs,
										error_convert_f=to_likelihood,
										verbose=verbose)

		else:
			raise Exception("method must be 'sgd', 'adam', or 'scg'")

		self.total_epochs += len(error_trace)
		self.error_trace += error_trace

		# Return neural network object to allow applying other methods
		# after training, such as:    Y = nnet.train(X, T, 100, 0.01).use(X)

		return self
    
	def _neg_log_likelihood_f(self, X, T):
		Y=self._forward(X)
		YLastLayer=Y[-1] # after forward X, we got the last layer of Y and ready to calculate LL(x)
		gs = self._softmax(YLastLayer) # gs=exp(Y@W)/rowSum(exp(Y@)W) See material 08
# 		LL = np.exp(-np.sum(T * np.log(gs)) / X.shape[0])
		LL=- np.mean(T * np.log(gs))
		return LL
 
	
	def _softmax(self, Y):
		'''Apply to final layer weighted sum outputs'''
		# Trick to avoid overflow
		maxY = Y.max()       
		expY = np.exp(Y - maxY)
		denom = expY.sum(1).reshape((-1, 1))
		Y = expY / (denom + sys.float_info.epsilon)
		return Y     
	
	def use(self, X):
		'''X assumed to not be standardized'''
		# Standardize X
		X = (X - self.X_means) / self.X_stds
		YLastLayer= self._forward(X)[-1]
		Ys = self._softmax(YLastLayer)

		predictedTrain = np.argmax(Ys,axis=1)
		
		return predictedTrain,Ys


	def _gradient_f(self, X, T):
		# Assumes forward_pass just called with layer outputs saved in self.Ys.
		n_samples = X.shape[0]
		n_outputs = T.shape[1]
		# D is delta matrix to be back propagated
		D = -(T - self._softmax(self.Ys[-1])) / (n_samples * n_outputs) # See material 09
		self._backpropagate(D)
		return self.all_gradients




#%% test function


# =============================================================================
# import neuralnetworks as nn
# 
# n = 500
# x1 = np.linspace(5, 20, n) + np.random.uniform(-2, 2, n)
# y1 = ((20-12.5)**2-(x1-12.5)**2) / (20-12.5)**2 * 10 + 14 + np.random.uniform(-2, 2, n)
# x2 = np.linspace(10, 25, n) + np.random.uniform(-2, 2, n)
# y2 = ((x2-17.5)**2) / (25-17.5)**2 * 10 + 5.5 + np.random.uniform(-2, 2, n)
# angles = np.linspace(0, 2*np.pi, n)
# x3 = np.cos(angles) * 15 + 15 + np.random.uniform(-2, 2, n)
# y3 = np.sin(angles) * 15 + 15 + np.random.uniform(-2, 2, n)
# X =  np.vstack((np.hstack((x1, x2, x3)),  np.hstack((y1, y2, y3)))).T
# T = np.repeat(range(1, 4), n).reshape((-1, 1))
# colors = ['blue', 'red', 'green']
# plt.figure(figsize=(6, 6))
# for c in range(1, 4):
#     mask = (T == c).flatten()
#     plt.plot(X[mask, 0], X[mask, 1], 'o', markersize=6,  alpha=0.5,  color=colors[c-1])
# 
# import mpl_toolkits.mplot3d as plt3
# from matplotlib import cm
# 
# nHidden = [5]
# nnet = nn.NeuralNetworkClassifier(2, nHidden, 3) # 3 classes, will actually make 2-unit output layer
# nnet.train(X, T, n_epochs=3000,  method='scg')
#  
# # print(Ws)
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
#  
# plt.subplot(2, 2, 2)
# 
# for c in range(1, 4):
#     mask = (T == c).flatten()
#     plt.plot(X[mask, 0], X[mask, 1], 'o', markersize=6,  alpha=0.5,  color=colors[c-1])
# 
# plt.subplot(2, 2, 4)
# plt.contourf(Xtest[:, 0].reshape((40, 40)), Xtest[:, 1].reshape((40, 40)),  predTest.reshape((40, 40))+1, 
#              levels = [0.5, 1.99, 2.01, 3.5],  #    levels=(0.5, 1.5, 2.5, 3.5), 
#              colors=colors);
# plt.show()
# =============================================================================
