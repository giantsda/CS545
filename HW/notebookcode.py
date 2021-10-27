#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Convolutional-Neural-Networks" data-toc-modified-id="Convolutional-Neural-Networks-1">Convolutional Neural Networks</a></span><ul class="toc-item"><li><span><a href="#Requirements" data-toc-modified-id="Requirements-1.1">Requirements</a></span></li></ul></li><li><span><a href="#Experiments" data-toc-modified-id="Experiments-2">Experiments</a></span></li><li><span><a href="#Grading" data-toc-modified-id="Grading-3">Grading</a></span></li><li><span><a href="#Extra-Credit" data-toc-modified-id="Extra-Credit-4">Extra Credit</a></span></li></ul></div>

# # Convolutional Neural Networks
# 
# For this assignment, use the `NeuralNetworkClassifier_CNN` class defined for you in `neuralnetworks_A4.py` contained in [A4code.tar](https://www.cs.colostate.edu/~anderson/cs545/notebooks/A4code.tar).  This tar file also includes other functions you will use here, contained in `mlfuncs.py`.

# In[100]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[101]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import neuralnetworks_A4 as nn
import mlfuncs


# ## Requirements

# First, look carefully at the `neuralnetworks_A4.py` and `optimizers.py` code provided above.  Some changes have been made in each. The most significant change is that the `train` function now accepts a `batch_size` argument so that the gradients we calculate don't have to be over the whole training set.  Recall that we can easily run out of memory with convolutional networks if we calculate gradients over the whole training set.  Also, `'scg'` is not a valid optimizer in this version of the code.
# 
# Implement the following functions:
# 
#     dataframe_result = run_these_parameters(X, T, n_folds,
#                                             layers_structs, 
#                                             methods, 
#                                             epochs, 
#                                             learning_rates.
#                                             batch_sizes)
#                                               
#     result = train_this_partition(Xtrain, Ttrain, Xval, Tval, Xtest, Ttest,
#                                   struct,
#                                   n_epochs, 
#                                   method, 
#                                   learning_rate,
#                                   batch_size)
#                                   
# The file `mlfuncs.py` contains several functions you will need to define these two required functions.  They are illustrated in the following examples.

# In[102]:


Y = np.array([0, 1, 1, 0, 0]).reshape(-1, 1)
T = np.array([0, 1, 0, 1, 0]).reshape(-1, 1)
mlfuncs.percent_equal(Y, T)


# The purpose of that one is obvious.  This next one is needed for storing your network stucture in a pandas DataFrame.  The structure must be an immutable data type.  A list is mutable, but a tuple is not.  So we must make sure all parts of the network structure specification is composed of tuples, not lists.

# In[103]:


struct = [ [], [10]]
mlfuncs.list_to_tuple(struct)


# In[104]:


struct = [ [[2, 4, 1], [5, 4, 2]], [20, 10]]
mlfuncs.list_to_tuple(struct)


# And here is a function that generates all training, validation, and testing partitions given the data and the number of folds.  It creates the partitions in a stratified manner, meaning all folds will have close to the same proportion of samples from each class.

# In[105]:


X = np.arange(12).reshape(6, 2)
T = np.array([0, 0, 1, 0, 1, 1]).reshape(-1, 1)
X, T


# In[106]:


for Xtrain, Ttrain, Xval, Tval, Xtest, Ttest in mlfuncs.generate_partitions(X, T, n_folds=3, classification=True):
        print(Xtrain, '\n', Ttrain, '\n', Xval, '\n', Tval, '\n', Xtest, '\n', Ttest)
        print()


# The function `run_these_parameters` loops through all values in `layers_structs`, `methods`, `epochs`, `learning rates` and `batch_sizes`.  For each set of parameter values, it loops through all ways of creating training, validation, and testing partitions using `n_folds`.  For each of these repetitions, `train_this_partition` is called to create the specified convolutional neural network, trains it, collects the percent correct on training, validation, and test sets, and returns a list of parameter values and the three accuracies.  `run_these_parameters` returns all of these results as a `pandas` DataFrame with column names `('struct', 'method', 'n_epochs', 'learning_rate', 'batch_size', 'train %', 'val %', 'test %')`. 
# 
# The resulting DataFrame results stored in variable `df` can be summarized with a statement like
# 
#       df.groupby(['struct', 'method', 'n_epochs', 'learning_rate',
#                   'batch_size']).mean())

# Define the two required functions in code cells above this cell.
# 
# The following examples show examples of how they should run, as

# In[107]:


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

n_each_class = 10
X, T = make_images(n_each_class)
p = 0
for i in range(4 * n_each_class):
    p += 1
    plt.subplot(4, n_each_class, p)
    plt.imshow(-X[i, :, :, 0], cmap='gray')
    plt.axis('off')


# In[108]:


n_each_class = 500
X, T = make_images(n_each_class)


# In[109]:


import pandas as pd



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
#                             print(f'Doing {struct};{method};{epoch};{learning_rate};{batch_size}')
                            Restult = train_this_partition(Xtrain, Ttrain, Xval, Tval, Xtest, Ttest, struct, epoch, method, learning_rate, batch_size)
                            resultData.append(Restult)

    table = pd.DataFrame(resultData, columns=classes)
    return table


# In[110]:


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
result


# In[111]:


# df = run_these_parameters(X, T, n_folds=4,
#                          structs=[
#                              [ [], [] ],
#                              [ [], [10] ],
#                              [[[5, 3, 1]], []],
#                              [[[20, 3, 2], [5, 3, 1]], [20]],
#                             ],
#                           methods=['adam'], # , 'sgd'],
#                           epochs=[10],
#                           learning_rates=[0.01], #, 0.1],
#                           batch_sizes=[3])
# df


# # Experiments
# 
# When you have `train_this_partition` and `run_these_parameters`, use them to explore the parameter values, trying to find combinations of parameter values that result in high validation accuracies.  
# 
# Start with one value for each of the five parameters, but remember to specifiy them as a list of one element, like `learning_rates=[0.01]`.  Then run again with 3 or 4 values for one parameter.  Note the best value.  Use that value for that parameter, then add more values for a different parameter.  
# 
# Proceed this way for each of the parameter values.  Discuss what you observe after each call to `run_these_parameters` with at least two sentences for each run.  Do the parameter values you find that work best surprise you?  Also discuss how well the validation and test accuracies equal each other.
# 
# For each method, try various hidden layer structures, learning rates, and numbers of epochs.  Use the validation percent accuracy to pick the best hidden layers, learning rates and numbers of epochs for each method.  Report training, validation and test accuracy for your best validation results for each of the three methods.
# 

# In[112]:


df = run_these_parameters(X, T, n_folds=4,
                         structs=[
                             [ [], [] ],
                             [ [], [10] ],
                             [[[5, 3, 1]], []],
                             [[[20, 3, 2], [5, 3, 1]], [20]],
                            ],
                          methods=['adam'], # , 'sgd'],
                          epochs=[10],
                          learning_rates=[0.01], #, 0.1],
                          batch_sizes=[3])
df


# (((5, 3, 1),), ())	is the best  for structs.
# The most complex structures does not produces the best results.
# In most case, result with train set is bette than test set. 

# In[113]:


df = run_these_parameters(X, T, n_folds=4,
                         structs=[
                             [[[5, 3, 1]], []],           
                            ],
                          methods=['adam' , 'sgd'],
                          epochs=[10],
                          learning_rates=[0.01], #, 0.1],
                          batch_sizes=[3])
df


# Results usign sgd and adam are compareable and all of them are acceptable.
# Sgd performs slightly better than adam

# In[82]:


df = run_these_parameters(X, T, n_folds=4,
                         structs=[
                             [[[5, 3, 1]], []],           
                            ],
                          methods=['sgd'],
                          epochs=[10,20,30],
                          learning_rates=[0.01], #, 0.1],
                          batch_sizes=[3])
df


# The results using 10,20,30 are comparable with 20 better than others.
# Probably because the batch size was set too small.

# In[84]:


df = run_these_parameters(X, T, n_folds=4,
                         structs=[
                             [[[5, 3, 1]], []],           
                            ],
                          methods=['sgd'],
                          epochs=[20],
                          learning_rates=[0.01,0.05,0.1], #, 0.1],
                          batch_sizes=[3])
df


# df = run_these_parameters(X, T, n_folds=4,
#                          structs=[
#                              [[[5, 3, 1]], []],           
#                             ],
#                           methods=['sgd'],
#                           epochs=[20],
#                           learning_rates=[0.01,0.05,0.1], #, 0.1],
#                           batch_sizes=[3])
# df

# With   learning_rates=[0.01,0.05,0.1], the rate=0.01 performance best.
# 

# In[86]:


df = run_these_parameters(X, T, n_folds=4,
                         structs=[
                             [[[5, 3, 1]], []],           
                            ],
                          methods=['sgd'],
                          epochs=[20],
                          learning_rates=[0.01], #, 0.1],
                          batch_sizes=[3,5,20,30])
df


# Tried batch_sizes=[3,5,20,30]), and looks like batch size=30 performs best.
# This is due to when batch size is sufficiently large, the pattern difference between the train set and the test set are reduced. 
# 
# 
# Do the parameter values you find that work best surprise you? Also discuss how well the validation and test accuracies equal each other.
# 
# The best parameter values I found out is  "
#                          structs=[
#                              [[[5, 3, 1]], []],           
#                             ],
#                           methods=['sgd'],
#                           epochs=[20],
#                           learning_rates=[0.01], #, 0.1],
#                           batch_sizes=[30])
# "
# It does not surprise me. The validation and test accuracies are acceptablely close to each other.

# df = run_these_parameters(X, T, n_folds=5,
#                          structs=[
#                              [ [], [] ],
#                              [ [], [10] ],
#                              [[[5, 3, 1]], []],
#                              [[[20, 3, 2], [5, 3, 1]], [20]], 
#                             ],
#                           methods=['sgd','adam'],
#                           epochs=[30],
#                           learning_rates=[0.01,0.05,0.1], #, 0.1],
#                           batch_sizes=[30])
# 
# 
# df

# For each method, try various hidden layer structures, learning rates, and numbers of epochs. Use the validation percent accuracy to pick the best hidden layers, learning rates and numbers of epochs for each method. Report training, validation and test accuracy for your best validation results for each of the three methods.
# 
# For meshod adam, the best combination happen to be 
# 
#   structs=[
#                            
#                              [[[5, 3, 1]], []],
#                               
#                             ],
#                            
#                           epochs=[20],
#                           learning_rates=[0.01],
#                           batch_sizes=[30])
# 
# The accuracy is training=100.00	validation=90.5	test=87.5
# 
# For method sgd, the best combination happen to be 
# 
#   structs=[ [[[20, 3, 2], [5, 3, 1]], [20]] ],                   
#                           epochs=[20],
#                           learning_rates=[0.01],
#                           batch_sizes=[30])
# 
# The accuracy is    training100	validation=92.5	test=85
# 
# 

# # Grading
# 
# (UPDATED Oct. 21, 9:35am, tolerance on accuracies is now larger) Download [A4grader.tar](https://www.cs.colostate.edu/~anderson/cs545/notebooks/A4grader.tar), extract `A4grader.py` before running the following cell.

# 

# In[116]:


get_ipython().run_line_magic('run', '-i A4grader.py')


# # Extra Credit
# 
# Repeat the above experiment using a convolutional neural network defined in `Pytorch`.  Implement this yourself by directly calling `torch.nn` functions.
