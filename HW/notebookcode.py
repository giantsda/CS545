#!/usr/bin/env python
# coding: utf-8

# # A1: Three-Layer Neural Network

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Requirements" data-toc-modified-id="Requirements-1">Requirements</a></span></li><li><span><a href="#Example-Results" data-toc-modified-id="Example-Results-2">Example Results</a></span></li><li><span><a href="#Discussion" data-toc-modified-id="Discussion-3">Discussion</a></span></li></ul></div>

# ## Requirements

# In this assignment, you will start with code from lecture notes 04 and add code to do the following.
# 
# * Add another hidden layer, for a total of two hidden layers.  This layer will use a weight matrix named `U`.  Its outputs will be named `Zu` and the outputs of the second hidden layer will be changed to `Zv`.
# * Define function `forward` that returns the output of all of the layers in the neural network for all samples in `X`. `X` is assumed to be standardized and have the initial column of constant 1 values.
# 
#       def forward(X, U, V, W):
#           .
#           .
#           .
#           Y = . . . # output of neural network for all rows in X
#           return Zu, Zv, Y
#       
# * Define function `gradient` that returns the gradients of the mean squared error with respect to each of the three weight matrices. `X` and `T` are assumed to be standardized and `X` has the initial column of 1's.
# 
#       def gradient(X, T, Zu, Zv, Y, U, V, W):
#           .
#           .
#           .
#           return grad_wrt_U, grad_wrt_V, grad_wrt_W
#           
# * Define function `train` that returns the resulting values of `U`, `V`, and `W` and the standardization parameters.  Arguments are unstandardized `X` and `T`, the number of units in the two hidden layers, the number of epochs and the learning rate, which is the same value for all layers. This function standardizes `X` and `T`, initializes `U`, `V` and `W` to uniformly distributed random values between -1 and 1, and `U`, `V` and `W` for `n_epochs` times as shown in lecture notes 04.  This function must call `forward`, `gradient` and `addOnes`.
# 
#       def train(X, T, n_units_U, n_units_V, n_epochs, rho):
#           .
#           .
#           .
#           return U, V, W, X_means, X_stds, T_means, T_stds
#           
# * Define function `use` that accepts unstandardized `X`, standardization parameters, and weight matrices `U`, `V`, and `W` and returns the unstandardized output.
# 
#       def use(X, X_means, X_stds, T_means, T_stds, U, V, W):
#           .
#           .
#           .
#           Y = ....
#           return Y

# ## Example Results

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def addOnes(X):
    return np.insert(X, 0, 1, axis=1)


# Add code cells here to define the functions above.  Once these are correctly defined, the following cells should run and produce similar results as those here.

# In[2]:


def rmse(T, Y, Tstds):
        error = (T - Y) * Tstds 
        return np.sqrt(np.mean(error ** 2))


# In[3]:


def forward(X, U, V, W):
    Zu= np.tanh(X @ U)
    Zv=np.tanh(addOnes(Zu) @ V)
    Z1 = addOnes(Zv)
    Y= Z1 @ W
    return Zu, Zv, Y


# In[4]:


def gradient(X, T, Zu, Zv, Y, U, V, W):
 
    Dw = T - Y
    grad_wrt_W=-addOnes(Zv).T@Dw
    Dv=Dw@W[1:,:].T*(1-Zv**2)
    grad_wrt_V=-addOnes(Zu).T@Dv
    Du=Dv@V[1:,:].T*(1-Zu**2)
    grad_wrt_U=-addOnes(X).T@Du
  
# 	grad_wrt_W=-addOnes(Zv).T@(T-Y)
# 	grad_wrt_V=-addOnes(Zu).T@(((T-Y)@W[1:, :].T)*(1-Zv**2))
# 	grad_wrt_U=-addOnes(X).T@(((T-Y)@W[1:, :].T)*(1-Zv**2)@V[1:, :].T*(1-Zu**2))
 
    
    return grad_wrt_U, grad_wrt_V, grad_wrt_W


# In[5]:


def use(X, X_means, X_stds, T_means, T_stds, U, V, W):
    X=(X-X_means)/X_stds
    Zu, Zv, Y=forward(addOnes(X), U, V, W)
    Y=Y*T_stds+T_means
    return Y 


# In[6]:


def train(X, T, n_units_U, n_units_V, n_epochs, rho):
    Xmeans = X.mean(axis=0)
    Xstds = X.std(axis=0)
    Tmeans = T.mean(axis=0)
    Tstds = T.std(axis=0)
    XtrainS = (X - Xmeans) / Xstds
    TtrainS = (T - Tmeans) / Tstds
    XtrainS1 = addOnes(XtrainS)

    U = np.random.uniform(-1, 1, size=(XtrainS1.shape[1], n_units_U))  
    V = np.random.uniform(-1, 1, size=(n_units_U + 1, n_units_V))  
    W = np.random.uniform(-1, 1, size=(n_units_V+1, TtrainS.shape[1]))  

    rhoI = rho/(T.shape[0]*T.shape[1])

    error=[]

    for epoch in range(n_epochs):
        Zu, Zv, Y=forward(XtrainS1, U, V, W)

        grad_wrt_U, grad_wrt_V, grad_wrt_W=gradient(XtrainS, TtrainS, Zu, Zv, Y, U, V, W)

        # Take step down the gradient
        U = U - rhoI * grad_wrt_U
        V = V - rhoI * grad_wrt_V  
        W = W - rhoI * grad_wrt_W  		

        error.append(rmse(TtrainS, Y, Tstds))
#         if epoch==95:
#             aaa=111
 

        if epoch%50==0:
            print(epoch)
 
    return U, V, W, Xmeans, Xstds, Tmeans, Tstds


# In[ ]:





# In[7]:


Xtrain = np.arange(4).reshape(-1, 1)
Ttrain = Xtrain ** 2

Xtest = Xtrain + 0.5
Ttest = Xtest ** 2


# In[8]:


U = np.array([[1, 2, 3], [4, 5, 6]])  # 2 x 3 matrix, for 2 inputs (include constant 1) and 3 units
V = np.array([[-1, 3], [1, 3], [-2, 1], [2, -4]]) # 2 x 3 matrix, for 3 inputs (include constant 1) and 2 units
W = np.array([[-1], [2], [3]])  # 3 x 1 matrix, for 3 inputs (include constant 1) and 1 ounit


# In[9]:


X_means = np.mean(Xtrain, axis=0)
X_stds = np.std(Xtrain, axis=0)
Xtrain_st = (Xtrain - X_means) / X_stds


# In[10]:


Zu, Zv, Y = forward(addOnes(Xtrain_st), U, V, W)
print('Zu = ', Zu)
print('Zv = ', Zv)
print('Y = ', Y)


# In[11]:


T_means = np.mean(Ttrain, axis=0)
T_stds = np.std(Ttrain, axis=0)
Ttrain_st = (Ttrain - T_means) / T_stds
grad_wrt_U, grad_wrt_V, grad_wrt_W = gradient(Xtrain_st, Ttrain_st, Zu, Zv, Y, U, V, W)
print('grad_wrt_U = ', grad_wrt_U)
print('grad_wrt_V = ', grad_wrt_V)
print('grad_wrt_W = ', grad_wrt_W)


# In[12]:


Y = use(Xtrain, X_means, X_stds, T_means, T_stds, U, V, W)
Y


# Here is another example that just shows the final results of training.

# In[13]:


n = 30
Xtrain = np.linspace(0., 20.0, n).reshape((n, 1)) - 10
Ttrain = 0.2 + 0.05 * (Xtrain + 10) + 0.4 * np.sin(Xtrain + 10) + 0.2 * np.random.normal(size=(n, 1))

Xtest = Xtrain + 0.1 * np.random.normal(size=(n, 1))
Ttest = 0.2 + 0.05 * (Xtest + 10) + 0.4 * np.sin(Xtest + 10) + 0.2 * np.random.normal(size=(n, 1))


# In[14]:


U, V, W, X_means, X_stds, T_means, T_stds = train(Xtrain, Ttrain, 5, 5, 100, 0.01)


# In[15]:


Y = use(Xtrain, X_means, X_stds, T_means, T_stds, U, V, W)


# In[16]:


plt.plot(Xtrain, Ttrain)
plt.plot(Xtrain, Y);


# In[17]:


U, V, W, X_means, X_stds, T_means, T_stds = train(Xtrain, Ttrain, 5, 5, 100000, 0.1)
Y = use(Xtrain, X_means, X_stds, T_means, T_stds, U, V, W)
plt.plot(Xtrain, Ttrain, label='Train')
plt.plot(Xtrain, Y, label='Test')
plt.legend();
plt.show()


# ## Discussion

# In this markdown cell, describe what difficulties you encountered in completing this assignment. What parts were easy for you and what parts were hard?

# # Grading
# 
# **A1grader.tar is now available.**
# 
# Your notebook will be run and graded automatically. Test this grading process by first downloading [A1grader.tar](http://www.cs.colostate.edu/~anderson/cs545/notebooks/A1grader.tar) and extract `A1grader.py` from it. Run the code in the following cell to demonstrate an example grading session.  The remaining 10 points will be based on your discussion of this assignment.
# 
# A different, but similar, grading script will be used to grade your checked-in notebook. It will include additional tests. You should design and perform additional tests on all of your functions to be sure they run correctly before checking in your notebook.  
# 
# For the grading script to run correctly, you must first name this notebook as 'Lastname-A1.ipynb' with 'Lastname' being your last name, and then save this notebook.

# In[ ]:


get_ipython().run_line_magic('run', '-i A1grader.py')


# # Check-In
# 
# Do not include this section in your notebook.
# 
# Name your notebook ```Lastname-A1.ipynb```.  So, for me it would be ```Anderson-A1.ipynb```.  Submit the file using the ```Assignment 1``` link on [Canvas](https://colostate.instructure.com/courses/131494).

# # Extra Credit
# 
# Apply your multilayer neural network code to a regression problem using data that you choose 
# from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets.php). Pick a dataset that
# is listed as being appropriate for regression.
