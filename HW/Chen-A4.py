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
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import pandas as pd
import neuralnetworks_A4 as nn
import mlfuncs


def make_images(n_each_class):
	'''Make 20x20 black and white images with diamonds or squares for the two classes, as line drawings.'''
	images = np.zeros((n_each_class * 4, 20, 20))  # nSamples, rows, columns
	radii = 3 + np.random.randint(10 - 5, size=(n_each_class * 4, 1))
	centers = np.zeros((n_each_class * 4, 2))
	for i in range(n_each_class * 4):
		r = radii[i, 0]
		centers[i, :] = r + 1 + np.random.randint(18 - 2 * r, size=(1, 2))
		x = int(centers[i, 0])
		y = int(centers[i, 1])
		if i < n_each_class:
			# plus
			images[i, x - r:x + r, y] = 1.0
			images[i, x, y - r:y + r] = 1.0
		elif i < n_each_class * 2:
			# minus
			images[i, x, y - r:y + r] = 1.0
		elif i < n_each_class * 3:
			# x
			images[i, range(x - r, x + r), range(y - r, y + r)] = 1.0
			images[i, range(x - r, x + r), range(y + r, y - r, -1)] = 1.0
		else:
			# /
			images[i, range(x - r, x + r), range(y - r, y + r)] = 1.0

	T = np.array(['plus'] * n_each_class + ['minus'] * n_each_class + ['times'] * n_each_class + ['divide'] * n_each_class).reshape(-1, 1)

	n, r, c = images.shape
	images = images.reshape(n, r, c, 1)  # add channel dimsension
	return images, T







# from A4mysolution import *
def train_this_partition(Xtrain, Ttrain, Xval, Tval, Xtest, Ttest, struct, n_epochs, method, learning_rate, batch_size):
    mlfuncs.make_batches(Xtrain, Ttrain, batch_size)
    nnet_cnn = nn.NeuralNetworkClassifier_CNN([Xtrain.shape[1], Xtrain.shape[2], Xtrain.shape[3]], struct[0], struct[1], np.unique(Ttrain))
    nnet_cnn.train(Xtrain, Ttrain, n_epochs, method=method, learning_rate=learning_rate, momentum=0.1, batch_size=batch_size, verbose=False)
    
    Yval,Y = nnet_cnn.use(Xval)
    Ytest,Y = nnet_cnn.use(Xtest)
    Ytrain,Y = nnet_cnn.use(Xtrain)
    
    percentageTrain = mlfuncs.percent_equal(Ytrain, Ttrain)
    percentageVal= mlfuncs.percent_equal(Yval, Tval)
    percentageTest = mlfuncs.percent_equal(Ytest, Ttest)
    Restult = []
#   ('struct', 'method', 'n_epochs', 'learning_rate', 'batch_size', 'train %', 'val %', 'test %').


    Restult.append(mlfuncs.list_to_tuple(struct))
    Restult.append(mlfuncs.list_to_tuple(method))
    Restult.append(mlfuncs.list_to_tuple(n_epochs))
    Restult.append(mlfuncs.list_to_tuple(learning_rate))
    Restult.append(mlfuncs.list_to_tuple(batch_size))
    Restult.append(mlfuncs.list_to_tuple(percentageTrain))
    Restult.append(mlfuncs.list_to_tuple(percentageVal))
    Restult.append(mlfuncs.list_to_tuple(percentageTest))
 
    return Restult



def run_these_parameters(X, T, n_folds,
            structs, 
            methods, 
            epochs, 
            learning_rates,
            batch_sizes):
    classes = ['struct', 'method', 'n_epochs', 'learning_rate', 'batch_size', 'train %', 'val %', 'test %']
    resultData = []
    for struct in structs:
        for method in methods:
            for epoch in epochs:
                for learning_rate in learning_rates:
                    for batch_size in batch_sizes:
#                           +struct, method, epoch, learning_rate, Xtest, batch_size
                        for Xtrain, Ttrain, Xval, Tval, Xtest, Ttest in mlfuncs.generate_partitions(X, T, n_folds, validation=True,shuffle=True, classification=True):
                            print(f'Doing {struct};{method};{epoch};{learning_rate};{batch_size}')
#                             Restult = train_this_partition(Xtrain, Ttrain, Xval, Tval, Xtest, Ttest, struct, epoch, method, learning_rate, batch_size)
#                             resultData.append(Restult)

    table = pd.DataFrame(resultData, columns=classes)
    return table


# =============================================================================
#  test
# =============================================================================


n_each_class = 10
X, T = make_images(n_each_class)
p = 0
# for i in range(4 * n_each_class):
# 	p += 1
# 	plt.subplot(4, n_each_class, p)
# 	plt.imshow(-X[i, :, :, 0], cmap='gray')
# 	plt.axis('off')
	

Y = np.array([0, 1, 1, 0, 0]).reshape(-1, 1)
T = np.array([0, 1, 0, 1, 0]).reshape(-1, 1)
mlfuncs.percent_equal(Y, T)
struct = [ [], [10]]
mlfuncs.list_to_tuple(struct)
struct = [ [[2, 4, 1], [5, 4, 2]], [20, 10]]
mlfuncs.list_to_tuple(struct)
X = np.arange(12).reshape(6, 2)
T = np.array([0, 0, 1, 0, 1, 1]).reshape(-1, 1)
X, T

for Xtrain, Ttrain, Xval, Tval, Xtest, Ttest in mlfuncs.generate_partitions(X, T, n_folds=3, classification=True):
	print(Xtrain, '\n', Ttrain, '\n', Xval, '\n', Tval, '\n', Xtest, '\n', Ttest)
	print()
	
n_each_class = 500
X, T = make_images(n_each_class)

struct = [ [[2, 5, 1]], [5] ]
n_epochs = 10
method= 'adam'
learning_rate = 0.01
batch_size = 10

n_samples = X.shape[0]
rows = np.arange(n_samples)
np.random.shuffle(rows)
ntrain = int(n_samples * 0.8)
nval = int(n_samples * 0.1)
Xtrain = X[rows[:ntrain], ...]
Ttrain = T[rows[:ntrain], ...]
Xval = X[rows[ntrain:ntrain+nval], ...]
Tval = T[rows[ntrain:ntrain+nval], ...]
Xtest = X[rows[ntrain+nval:], ...]
Ttest = T[rows[ntrain+nval:], ...]
		   
result = train_this_partition(Xtrain, Ttrain, Xval, Tval, Xtest, Ttest,
							  struct, n_epochs, method, learning_rate, batch_size)
 

df = run_these_parameters(X, T, n_folds=2,
                         structs=[
                             [],
                             [ [], [] ],
                             [ [], [10] ],
                             [[[5, 3, 1]], []],
                             [[[20, 3, 2], [5, 3, 1]], [20]],
                            ],
                          methods=['adam' , 'sgd'],
                          epochs=[10],
                          learning_rates=[0.01], #, 0.1],
                          batch_sizes=[3])
df






















