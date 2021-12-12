import numpy as np
import optimizers as opt
import sys  # for sys.float_info.epsilon
import matplotlib.pyplot as plt



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
		self.shapes=shapes
		self.epoch=0

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
		self.X_std=X
		self.T_std=T
		
		
		# Instantiate Optimizers object by giving it vector of all weights
		optimizer = opt.Optimizers(self.all_weights)

		_error_convert_f = lambda err: (np.sqrt(err) * self.T_stds)[0]
		
		if method == 'adadelta':
			error_trace=self.adadelta(self._forward,self._gradient_f,n_epochs,learning_rate)
		elif method == 'adadeltaAdapt':
			error_trace=self.adadeltaAdapt(self._forward,self._gradient_f,n_epochs,learning_rate)
		elif method == 'adm':
			error_trace=self.adm_chen (self.forward_ADM, self.all_weights.size, self.all_weights, 1e-12, n_epochs,0.9999,3)
		elif method == 'sgd':
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
	
	def adadelta(self, function, gradient, n_iter, rho):
		# Based on https://d2l.ai/chapter_optimization/adadelta.html
		ep=1e-5
		solutions = list()
		size=self.all_weights.size
		solution = np.random.uniform(-1, 1, size=size)
		St = np.zeros(shape=size)
		delta = np.zeros(shape=size)

		error_trace = []
		# Main loop
		for iterationI in range(n_iter):
			error=self._error_f(self.X_std,self.T_std)
			error=np.sqrt(error) * self.T_stds
		
			print("iteration %d, error=%f" % (iterationI,error))
			dW = gradient(self.X_std, self.T_std).reshape(size,)
			St=St* rho + (dW**2.0 * (1.0-rho))
			rescaledGradient = np.sqrt(delta+ep)/np.sqrt(St+ep)*dW
			solution = solution - rescaledGradient
			delta=delta * rho + rescaledGradient**2.0 * (1.0-rho)
			
			self.all_weights=solution
					# unpack self.all_weights to self.Ws
			i=0
			for layerI in range(len(self.shapes)):
				shapeX=self.shapes[layerI][0]
				shapeY=self.shapes[layerI][1]
				self.Ws[layerI]=self.all_weights[i:i+shapeX*shapeY].reshape(shapeX,shapeY)
				i+=shapeX*shapeY
				
			solutions.append(solution)
			error_trace.append(error)
# 			print('>%d f(%s) = %.5f' % (it, solution, function(solution[0], solution[1])))
		return error_trace

	def adadeltaAdapt(self, function, gradient, n_iter, rho):
		# Based on https://d2l.ai/chapter_optimization/adadelta.html
		ep=1e-5
		solutions = list()
		size=self.all_weights.size
		solution = np.random.uniform(-1, 1, size=size)
		St = np.zeros(shape=size)
		delta = np.zeros(shape=size)

		error_trace = []
		# Main loop
		for iterationI in range(2,n_iter):
			error=self._error_f(self.X_std,self.T_std)
			error=np.sqrt(error) * self.T_stds
			if iterationI % 100==0:
				print("iteration %d, error=%f, rho=%f" % (iterationI,error,rho))
			dW = gradient(self.X_std, self.T_std).reshape(size,)
			St=St* rho + (dW**2.0 * (1.0-rho))
			rescaledGradient = np.sqrt(delta+ep)/np.sqrt(St+ep)*dW
			solution = solution - rescaledGradient
			delta=delta * rho + rescaledGradient**2.0 * (1.0-rho)
			
			self.all_weights=solution
					# unpack self.all_weights to self.Ws
			i=0
			for layerI in range(len(self.shapes)):
				shapeX=self.shapes[layerI][0]
				shapeY=self.shapes[layerI][1]
				self.Ws[layerI]=self.all_weights[i:i+shapeX*shapeY].reshape(shapeX,shapeY)
				i+=shapeX*shapeY
				
			solutions.append(solution)
			error_trace.append(error)
			if iterationI % int(n_iter/7)==0:
				rho=1-(1-rho)*0.1
# 			print('>%d f(%s) = %.5f' % (it, solution, function(solution[0], solution[1])))
		return error_trace
	
	def _forward(self, X):
		'''
		X assumed to be standardized and with first column of 1's
		'''
		self.Ys = [X]
		for W in self.Ws[:-1]:  # forward through all but last layer
			self.Ys.append(np.tanh(self.Ys[-1] @ W[1:, :] + W[0:1, :]))
		last_W = self.Ws[-1]
		self.Ys.append(self.Ys[-1] @ last_W[1:, :] + last_W[0:1, :])
		return self.Ys

	# Function to be minimized by optimizer method, mean squared error
	def _error_f(self, X, T):
		Ys = self._forward(X)
		mean_sq_error = np.mean((T - Ys[-1]) ** 2)




		if (self.epoch % 100)==0:
			plt.plot(self.Ys[-1], 'o-', label='Model ')
			plt.plot(self.T_std, '*-', label='Train')
			plt.draw()
			plt.pause(0.00001)
			plt.clf()
		self.epoch=self.epoch+1
				
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

	def adm_chen (self,F, n, x_old, tol, maxIteration,lmd_in,m_in):
		found=0
		kGlobal = 0
		X_end = x_old
		error_s=np.zeros(shape=(1,maxIteration))
		Y = F(X_end)
		err=max(abs(Y))
		print('adm_chen:n=%d Before solving, error=%2.15f \n' % (n,err))
		error_trace = []

		while (err>tol and kGlobal<=maxIteration):
			[X_end, kLocal,error]=self.mixing(F,n,X_end,tol,kGlobal,maxIteration,lmd_in,m_in)
			error_trace.append(error)
			
			kGlobal=kGlobal+kLocal
			Y = F(X_end)
			err=max(abs(Y))
	
		if err<tol:
			found=1
		else:
			print("And_chen failed after %d iterations :(  Try to increase max iteration allowed\n" % (maxIteration));
	
		x_old=X_end
		self.all_weights=x_old
		
		error_trace = [item for sublist in error_trace for item in sublist]
		return error_trace
 
	
	def mixing(self,F,n,x_old,tol,kGlobal,maxIteration,lmd_in,m_in):
		lmd = lmd_in
		lk = lmd
		nm=m_in
		kLocal = 0
		X=np.zeros((n,maxIteration))
		Y=np.zeros((n,maxIteration))
		X[:,kLocal] = x_old.ravel()
		err=1e99
		U=1
		error=[]
		while (err>tol and kLocal+kGlobal-1<maxIteration):
	
			Y[:,kLocal] = F(X[:,kLocal]).ravel()
			err=max(abs(Y[:,kLocal]))
			if (kLocal+kGlobal>=1):
				error.append(err)
			
			if kLocal>0: # and (kLocal%10)==0):
				print('adm iteration: %d,n=%d, lk=%e, error: %.14e\n' % (kGlobal+kLocal-1,n,lk,err))
			
			if (err < tol):
				found = 1;
				X_end = X[:,kLocal]
				print('*****And_chen: Solved equation successfully!*****\nThe solution is:\n');
				print(X[:,kLocal])
				return (X_end, kLocal,error)
	
			if (err > 1e7):
				explode=1
	
	# Calculate the matrix U and the column vector v
			if (kLocal <= nm):
				m = kLocal
			else:
				m = nm
			
			
			U=np.zeros((m,m))
			V=np.zeros((m,1))
			for i in range (0,m):
				V[i,0] = np.dot(Y[:,kLocal] - Y[:,kLocal-i-1],Y[:,kLocal])
				for j in range(m):
					U[i,j] =np.dot(Y[:,kLocal] - Y[:,kLocal-i-1],Y[:,kLocal] - Y[:,kLocal-j-1])
	 
	#Calculate c = U^(-1) * v using Gauss
			
			if (m > 0):
				if np.linalg.cond(U) < 1/1e-14:
					c =np.linalg.solve(U,V) 
				else:
					print("And_chen: Singular Matrix detected And_chen restarted!\n");
					X_end=X[:,kLocal]
					return (X_end, kLocal,error)
	
	# Calculate the next x^(k)
			for i in range (n):
				cx = 0
				cd = 0
				for j in range(m):
					cx = cx + c[j] * (X[i,kLocal-j-1] - X[i,kLocal])
					cd = cd + c[j] * (Y[i,kLocal-j-1] - Y[i,kLocal])
					
				X[i,kLocal+1] = X[i,kLocal] + cx + (1-lk)*(Y[i,kLocal]+cd)
			
			
			kLocal = kLocal + 1
			if (err<0.03 and kLocal+kGlobal-1>200):  # only modifiy lk if it is close to solution
				lk = lk * lmd
	
			
			if (lk<0.0001):  # reset lk if it is too small
				lk = lmd
	
		X_end=X[:,kLocal]
		return (X_end, kLocal,error)
	
	def forward_ADM(self, weights):
		X=self.X_std
		T=self.T_std
		if T.size>weights.size: #not sure if this will ever happen
			weights=np.append(weights,np.zeros(shape=[T.size-weights.size,1]))
			print("T.size is larger than weights.size. check how this happened")
		# unpack self.all_weights to self.Ws
		i=0
		for layerI in range(len(self.shapes)):
			shapeX=self.shapes[layerI][0]
			shapeY=self.shapes[layerI][1]
			self.Ws[layerI]=weights[i:i+shapeX*shapeY].reshape(shapeX,shapeY)
			i+=shapeX*shapeY
		
		self.Ys=[]
		for layerI in range(len(self.n_hidden_units_by_layers)):
			X=np.tanh(self.addOnes(X) @ self.Ws[layerI])
			self.Ys.append(X)
		X=self.addOnes(X)@self.Ws[-1]
		self.Ys.append(X)
		error = (T - self.Ys[-1]) * self.T_stds 
		#	very possible that the x and f(x) are not with the same length. Need to fill them with zeros.
		if error.size<weights.size:
			error=np.append(error,np.zeros(shape=[weights.size-error.size,1]))
			
# 		plt.plot(self.Ys[-1], 'o-', label='Model ')
# 		plt.plot(T, '*-', label='Train')

# 		plt.draw()
# 		plt.pause(0.00001)
# 		plt.clf()
		return error
	
	def addOnes(self,A):
		return np.insert(A, 0, 1, axis=1)
#%% main functions
np.random.seed(1)


X = np.arange(-2, 2, 0.05).reshape(-1, 1)
T = np.sin(X) * np.sin(X * 10)



nnet = NeuralNetwork(X.shape[1], [60,60], 1)
nnet.train(X, T, 100, method='adm', learning_rate=0.99)
Y = nnet.use(X)

plt.plot(nnet.error_trace, label='error')
exit()



adadelta=1






