#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

from IPython.display import display, clear_output  # for the following animation


def run(rho, n_epochs, stepsPerFrame=10):

	# Initialize weights to all zeros
	# For this demonstration, we will have one variable input. With the constant 1 input, we have 2 weights.
	W = np.zeros((2,1))

	# Collect the weights after each update in a list for later plotting. 
	# This is not part of the training algorithm
	ws = [W.copy()]

	# Create a bunch of x values for plotting
	xs = np.linspace(0, 10, 100).reshape((-1, 1))
	xs1 = np.insert(xs, 0, 1, axis=1)

	fig = plt.figure(figsize=(8, 8))

	# For each pass (one epoch) through all samples ...
	for iter in range(n_epochs):
		# For each sample ...
		for n in range(n_samples):
		
			# Calculate prediction using current model, w.
			#    n:n+1 is used instead of n to preserve the 2-dimensional matrix structure
			Y = X1[n:n + 1,:] @ W
			
			# Update w using negative gradient of error for nth sample
			W += rho * X1[n:n + 1, :].T * (T[n:n + 1, :] - Y)
			
			# Add new w to our list of past w values for plotting
			ws.append(W.copy())
			haha=n_samples+5
			print(haha)
		
			if n % stepsPerFrame == 0:
				fig.clf()

				# Plot the X and T data.
				plt.subplot(2, 1, 1)
				plt.plot(X, T, 'o', alpha=0.6, label='Data')
				plt.plot(X[n,0], T[n], 'ko', ms=10, label='Last Trained Sample')

				# Plot the output of our linear model for a range of x values
				plt.plot(xs, xs1 @ W, 'r-', linewidth=5, label='Model')
				plt.xlabel('$x$')
				plt.legend(loc='upper right')
				plt.xlim(0, 10)
				plt.ylim(0, 5)

				# In second panel plot the weights versus the epoch number
				plt.subplot(2, 1, 2)
				plt.plot(np.array(ws)[:, :, 0])
				plt.xlabel('Updates')
				plt.xlim(0, n_epochs * n_samples)
				plt.ylim(-1, 3)
				plt.legend(('$w_0$', '$w_1$'))
				plt.pause(0.001)
				clear_output(wait=True)
				display(fig)

				
	clear_output(wait=True)
	
	return W


n_samples = 100
X = np.random.uniform(0, 10, (n_samples, 1))
T = 2 - 0.1 * X + 0.05 * (X - 6)**2 + np.random.normal(0, 0.1, (n_samples,1))


plt.plot(X, T)

plt.plot(X, T, 'o');  # ;  suppresses one line of output like [<matplotlib.lines.Line2D at 0x7f9657fa0c40>]

X1 = np.insert(X, 0, 1, axis=1)

X1.shape, T.shape

X1[:5, :]

learning_rate = 0.01
n_samples = X1.shape[0]  # number of rows in data equals the number of samples

W = np.zeros((2, 1))                # initialize the weights to zeros
for epoch in range(10):             # train for this many epochs, or passes through the data set
	for n in range(n_samples):
		haha= X1[n:n + 1, :]
		Y = X1[n:n + 1, :] @ W      # predicted value, y, for sample n
		error = (T[n:n + 1, :] - Y)  # negative gradient of squared error
		
		# update weights by fraction of negative derivative of square error with respect to weights
		W -=  learning_rate * -2 * X1[n:n + 1, :].T * error  
		
print(W)

plt.plot(X, T, 'o', label='Data')
plt.plot(X, X1 @ W, 'ro', label='Predicted')
plt.legend();
plt.show()




# run(0.01, n_epochs=1, stepsPerFrame=1)

def make_powers(X, max_power):
	return np.hstack([X ** p for p in range(1, max_power + 1)])


def train(X, T, n_epochs, rho):
	
	means = X.mean(0)
	stds = X.std(0)
	# Replace stds of 0 with 1 to avoid dividing by 0.
	stds[stds == 0] = 1
	Xst = (X - means) / stds
	
	Xst = np.insert(Xst, 0, 1, axis=1)  # Insert column of 1's as first column in Xst
	
	# n_samples, n_inputs = Xst.shape[0]
	n_samples, n_inputs = Xst.shape
	
	# Initialize weights to all zeros
	W = np.zeros((n_inputs, 1))  # matrix of one column
	
	# Repeat updates for all samples for multiple passes, or epochs,
	for epoch in range(n_epochs):
		
		# Update weights once for each sample.
		for n in range(n_samples):
		
			# Calculate prediction using current model, w.
			#    n:n+1 is used instead of n to preserve the 2-dimensional matrix structure
			Y = Xst[n:n + 1, :] @ W
			
			# Update w using negative gradient of error for nth sample
			W += rho * Xst[n:n + 1, :].T * (T[n:n + 1, :] - Y)
				
	# Return a dictionary containing the weight matrix and standardization parameters.
	return {'W': W, 'means' : means, 'stds' :stds, 'max_power': max_power}

def use(model, X):
	Xst = (X - model['means']) / model['stds']
	Xst = np.insert(Xst, 0, 1, axis=1)
	Y = Xst @ model['W']
	return Y

def rmse(A, B):
	return np.sqrt(np.mean( (A - B)**2 ))





n_samples = 40
training_fraction = 0.8
n_models = 1000
confidence = 90 # percent
max_power = 1  # linear model

X = np.hstack((np.linspace(0, 3, num=n_samples),
			   np.linspace(6, 10, num=n_samples))).reshape(2 * n_samples, 1)
T = -1 + 0 * X + 0.1 * X**2 - 0.02 * X**3 + 0.5 * np.random.normal(size=(2 * n_samples, 1))
X.shape, T.shape



plt.plot(X, T, '.-');
n_rows = X.shape[0]
row_indices = np.arange(n_rows)
np.random.shuffle(row_indices)
n_train = round(n_rows * training_fraction)
n_test = n_rows - n_train

Xtrain = X[row_indices[:n_train], :]
Ttrain = T[row_indices[:n_train], :]
Xtest = X[row_indices[n_train:], :]
Ttest = T[row_indices[n_train:], :]

Xtrain.shape, Ttrain.shape, Xtest.shape, Ttest.shape

plt.plot(Xtrain[:, 0], Ttrain, 'o', label='Train')
plt.plot(Xtest[:, 0], Ttest, 'ro', label='Test')
plt.legend(loc='best');

max_power = 1
Xtrain = X[row_indices[:n_train], :]
Xtest = X[row_indices[n_train:], :]
Xtrain = make_powers(Xtrain, max_power)
Xtest = make_powers(Xtest, max_power)

n_epochs = 1000
rho = 0.01

n_models = 10

models = []
for model_i in range(n_models):
	train_rows = np.random.choice(list(range(n_train)), n_train)
	Xtrain_boot = Xtrain[train_rows, :]
	Ttrain_boot = Ttrain[train_rows, :]
	model = train(Xtrain_boot, Ttrain_boot, n_epochs, rho)
	models.append(model)
	
	
	
Y_all = []
for model in models:
	Y_all.append( use(model, Xtest) )   
	
Y_all = np.array(Y_all).squeeze().T  # I like putting each model's output in a column, so `Y_all` now has each model's output for a sample in a row.
Ytest = np.mean(Y_all, axis=1)


RMSE_test = np.sqrt(np.mean((Ytest - Ttest)**2))
print(f'Test RMSE is {RMSE_test:.4f}')

n_plot = 200
Xplot = np.linspace(0, 12.5, n_plot).reshape(n_plot, 1)
Xplot_powers = make_powers(Xplot, max_power)
Ys = []
for model in models:
	Yplot = use(model, Xplot_powers)
	Ys.append(Yplot)

Ys = np.array(Ys).squeeze().T
Ys.shape


plt.figure(figsize=(10, 10))
plt.plot(Xtrain[:, 0], Ttrain, 'o')
plt.plot(Xtest[:, 0], Ttest, 'o')
plt.plot(Xplot, Ys, alpha=0.5);
plt.ylim(-14, 2);

max_power = 6
Xtrain = X[row_indices[:n_train], :]
Xtest = X[row_indices[n_train:], :]
Xtrain = make_powers(Xtrain, max_power)
Xtest = make_powers(Xtest, max_power)

n_epochs = 2000
rho = 0.05

n_models = 100 

models = []
for model_i in range(n_models):
	train_rows = np.random.choice(list(range(n_train)), n_train)
	Xtrain_boot = Xtrain[train_rows, :]
	Ttrain_boot = Ttrain[train_rows, :]
	model = train(Xtrain_boot, Ttrain_boot, n_epochs, rho)
	models.append(model)
	print(f'Model {model_i}', end=' ')
	
n_plot = 200
Xplot = np.linspace(0, 12.5, n_plot).reshape(n_plot, 1)
Xplot_powers = make_powers(Xplot, max_power)
Ys = []
for model in models:
	Yplot = use(model, Xplot_powers)
	Ys.append(Yplot)

Ys = np.array(Ys).squeeze().T

plt.figure(figsize=(10, 10))
plt.plot(Xtrain[:, 0], Ttrain, 'o')
plt.plot(Xtest[:, 0], Ttest, 'o')
plt.plot(Xplot, Ys, alpha=0.5);
plt.ylim(-14, 2);   

all_Ws = [model['W'] for model in models]
len(all_Ws), all_Ws[0].shape


all_Ws = np.array(all_Ws).squeeze()
all_Ws.shape



all_Ws = all_Ws[:, 1:]
all_Ws.shape


all_Ws = np.sort(all_Ws, axis=0)
low_high = all_Ws[[9, 89], :].T
low_high



for i, row in enumerate(low_high):
	print(f'Power {i + 1:2} Low {row[0]:6.2f} High {row[1]:6.2f}')

!curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data
!head yacht_hydrodynamics.data

data = np.loadtxt('yacht_hydrodynamics.data')

T = data[:, -1:]
X = data[:, :-1]
Xnames = ['Center of Buoyancy', 'Prismatic coefficient', 'Length-displacement ratio', 'Beam-draught ratio',
		  'Length-beam ratio', 'Froude number']
Tname = 'Resistance'
X.shape, T.shape, Xnames, Tname

plt.figure(figsize=(10, 10))
for i in range(6):
	plt.subplot(2, 3, i + 1)
	plt.plot(X[:, i] ,T, '.')
	plt.ylabel(Tname)
	plt.xlabel(Xnames[i])
plt.tight_layout()

plt.plot(X[:100, :])
plt.plot(T[:100, :])


model = train(X, T, n_epochs=1000, rho=0.01)
predict = use(model, X)
print(rmse(predict, T))


plt.plot(T)
plt.plot(predict)

plt.plot(T, predict, 'o')
plt.plot([0, 50], [0, 50],  'r-')
plt.xlabel('actual')
plt.ylabel('predicted')

plt.plot(X[:,-1], T, 'o');


plt.plot(X[:,-1]**2, T, 'o');

plt.plot(X[:,-1]**4, T, 'o');

plt.plot(X[:,-1]**8, T, 'o');

Xp = make_powers(X, 5)
model = train(Xp, T,  n_epochs=1000, rho=0.01)
predict = use(model, Xp)
print(rmse(predict, T))

plt.plot(T)
plt.plot(predict);

plt.plot(T, predict, 'o')
plt.plot([0, 50], [0, 50]);

n = 50
plt.plot(T[:n])
plt.plot(predict[:n]);

result = []
for max_power in range(1, 20):
	Xp = make_powers(X, max_power)
	model = train(Xp, T, n_epochs=1000, rho=0.001)
	error = rmse(use(model, Xp), T)
	print(f'{max_power=} {error=}')
	result.append([max_power, error])
result = np.array(result)
result

plt.plot(result[:,0],result[:,1],'o-')
plt.xlabel('Exponent of X')
plt.ylabel('RMSE')

Xp = make_powers(X, 6)
predict = use(train(Xp, T, n_epochs=1000, rho=0.01), Xp)

plt.plot(T)
plt.plot(predict)

plt.plot(T, predict, 'o')
plt.plot([0, 50], [0, 50])
plt.xlabel('Actual')
plt.ylabel('Predicted');







