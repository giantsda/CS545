#!/usr/bin/env python
# coding: utf-8

# # A2: NeuralNetwork Class

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Requirements" data-toc-modified-id="Requirements-1">Requirements</a></span></li><li><span><a href="#Code-for-NeuralNetwork-Class" data-toc-modified-id="Code-for-NeuralNetwork-Class-2">Code for <code>NeuralNetwork</code> Class</a></span></li><li><span><a href="#Example-Results" data-toc-modified-id="Example-Results-3">Example Results</a></span></li><li><span><a href="#Application-to-Boston-Housing-Data" data-toc-modified-id="Application-to-Boston-Housing-Data-4">Application to Boston Housing Data</a></span></li></ul></div>

# ## Requirements

# In this assignment, you will complete the implementation of the `NeuralNetwork` class, starting with the code included in the next code cell.  Your implementation must meet the requirements described in the doc-strings.
# 
# Download [optimizers.tar](https://www.cs.colostate.edu/~anderson/cs545/notebooks/optimizers.tar) and extract `optimizers.py` for use in this assignment.
# 
# Then apply your `NeuralNetwork` class to the problem of predicting the value of houses in Boston as described below.

# ## Code for `NeuralNetwork` Class

# In[ ]:


get_ipython().run_cell_magic('writefile', 'neuralnetwork.py', '# -*- coding: utf-8 -*-\n"""\nCreated on Mon Sep 13 09:55:02 2021\n\n@author: chen\n"""\nimport numpy as np\nimport matplotlib.pyplot as plt\n# %matplotlib inline\nimport time  # for sleep\nimport IPython.display as ipd  # for display and clear_output\nfrom IPython.display import display, clear_output  # for the following animation\nimport os\nimport copy\nimport signal\nimport os\nimport numpy as np\nfrom mpl_toolkits.mplot3d import Axes3D\nfrom matplotlib.colors import LightSource\nimport optimizers as opt\nimport pandas  # for reading csv file\n\n\n\n\n\nclass NeuralNetwork():\n    """\n    A class that represents a neural network for nonlinear regression\n\n    Attributes\n    ----------\n    n_inputs : int\n        The number of values in each sample\n    n_hidden_units_by_layers: list of ints, or empty\n        The number of units in each hidden layer.\n        Its length specifies the number of hidden layers.\n    n_outputs: int\n        The number of units in output layer\n    all_weights : one-dimensional numpy array\n        Contains all weights of the network as a vector\n    Ws : list of two-dimensional numpy arrays\n        Contains matrices of weights in each layer,\n        as views into all_weights\n    all_gradients : one-dimensional numpy array\n        Contains all gradients of mean square error with\n        respect to each weight in the network as a vector\n    Grads : list of two-dimensional numpy arrays\n        Contains matrices of gradients weights in each layer,\n        as views into all_gradients\n    total_epochs : int\n        Total number of epochs trained so far\n    error_trace : list\n        Mean square error (standardized) after each epoch\n    X_means : one-dimensional numpy array\n        Means of the components, or features, across samples\n    X_stds : one-dimensional numpy array\n        Standard deviations of the components, or features, across samples\n    T_means : one-dimensional numpy array\n        Means of the components of the targets, across samples\n    T_stds : one-dimensional numpy array\n        Standard deviations of the components of the targets, across samples\n        \n    Methods\n    -------\n    make_weights_and_views(shapes)\n        Creates all initial weights and views for each layer\n\n    train(X, T, n_epochs, method=\'sgd\', learning_rate=None, verbose=True)\n        Trains the network using samples by rows in X and T\n\n    use(X)\n        Applies network to inputs X and returns network\'s output\n    """ \n    def __init__(self, n_inputs, n_hidden_units_by_layers, n_outputs):\n\n        \n        self.n_inputs = n_inputs\n        self.n_outputs = n_outputs\n        self.n_hidden_units_by_layers = n_hidden_units_by_layers\n        self.Ws= []\n        self.all_weights=np.empty([0,1])\n        self.shapes=[]\n \n    \n        layer_n=n_inputs\n        for layerI in range(len(self.n_hidden_units_by_layers)):\n            layerI_N=self.n_hidden_units_by_layers[layerI]\n            self.shapes.append([1 + layer_n, layerI_N])\n            layer_n=layerI_N\n        self.shapes.append([1 + layer_n, n_outputs])\n        \n        self.make_weights_and_views(self.shapes)\n        self.all_gradients = []\n        self.Grads = []\n        self.total_epochs = 0\n        self.error_trace = []\n        self.X_means = None\n        self.X_stds = None\n        self.T_means = None\n        self.T_stds = None\n        self.Ys=None\n\n    def make_weights_and_views(self, shapes):\n        """Creates vector of all weights and views for each layer\n\n        Parameters\n        ----------\n        shapes : list of pairs of ints\n            Each pair is number of rows and columns of weights in each layer\n\n        Returns\n        -------\n        Vector of all weights, and list of views into this vector for each layer\n        """\n\n        # Create one-dimensional numpy array of all weights with random initial values\n        #  ...\n\n        # Build list of views by reshaping corresponding elements\n        # from vector of all weights into correct shape for each layer.        \n        # ...\n \n        for layerI in range(len(shapes)):\n            shapeX=shapes[layerI][0]\n            shapeY=shapes[layerI][1]\n            wI=1 / np.sqrt(shapeX) * np.random.uniform(-1, 1, size=(shapeX, shapeY))\n            self.Ws.append(wI)\n#           haha=np.vstack((self.all_weights,wI.reshape(-1,1)))\n            self.all_weights=np.vstack((self.all_weights,wI.reshape(-1,1))) \n \n    def __repr__(self):\n        return f\'NeuralNetwork({self.n_inputs}, \' + \\\n            f\'{self.n_hidden_units_by_layers}, {self.n_outputs})\'\n\n    def __str__(self):\n        s = self.__repr__()\n        if self.total_epochs > 0:\n            s += f\'\\n Trained for {self.total_epochs} epochs.\'\n            s += f\'\\n Final standardized training error {self.error_trace[-1]:.4g}.\'\n        return s\n \n    def train(self, X, T, n_epochs, method=\'sgd\', learning_rate=None, verbose=True):\n        """Updates the weights \n        Parameters\n        ----------\n        X : two-dimensional numpy array\n            number of samples  x  number of input components\n        T : two-dimensional numpy array\n            number of samples  x  number of output components\n        n_epochs : int\n            Number of passes to take through all samples\n        method : str\n            \'sgd\', \'adam\', or \'scg\'\n        learning_rate : float\n            Controls the step size of each update, only for sgd and adam\n        verbose: boolean\n            If True, progress is shown with print statements\n        """\n\n        # Calculate and assign standardization parameters\n        # ...\n        \n        self.X_means = np.mean(X, axis=0)\n        self.X_stds = np.std(X, axis=0)\n        self.T_means = np.mean(T, axis=0)\n        self.T_stds = np.std(T, axis=0)\n        # Standardize X and T\n\n        X = (X - self.X_means) / self.X_stds\n        T = (T - self.T_means) / self.T_stds\n \n        # Instantiate Optimizers object by giving it vector of all weights\n        optimizer = opt.Optimizers(self.all_weights)\n\n        error_convert_f = lambda err: (np.sqrt(err) * self.T_stds)[0]\n        \n        # Call the requested optimizer method to train the weights.\n\n        if method == \'sgd\':\n            error_trace=optimizer.sgd(self.error_f, self.gradient_f,fargs=[X,T],error_convert_f=error_convert_f,learning_rate=learning_rate,n_epochs=n_epochs,verbose=True)\n        elif method == \'adam\':\n            error_trace=optimizer.adam(self.error_f, self.gradient_f,fargs=[X,T],error_convert_f=error_convert_f,learning_rate=learning_rate,n_epochs=n_epochs)\n        elif method == \'scg\':\n            error_trace=optimizer.scg(self.error_f, self.gradient_f,fargs=[X,T],error_convert_f=error_convert_f,n_epochs=n_epochs)\n        else:\n            raise Exception("method must be \'sgd\', \'adam\', or \'scg\'")\n \n    \n        self.total_epochs += len(error_trace)\n        self.error_trace += error_trace\n\n\n\n        self._forward(X)\n        error = (T - self.Ys[-1]) * self.T_stds \n#       plt.plot(X, self.Ys[-1], \'o-\', label=\'Model \')\n#       errors.append(nnet.get_error_trace())\n#       plt.plot(X, T, \'*-\', label=\'Train\')\n#       print(self.all_weights[20])\n#       plt.plot(self.all_weights, \'*-\', label=\'w\')\n\n#       plt.show()\n\n#       plt.draw()\n#       plt.pause(0.00000001)\n#       plt.clf()\n \n        \n        \n        \n        \n        # Return neural network object to allow applying other methods\n        # after training, such as:    Y = nnet.train(X, T, 100, 0.01).use(X)\n\n        return self\n\n    def addOnes(self,A):\n        return np.insert(A, 0, 1, axis=1)\n    \n    def _forward(self, X):\n        """Calculate outputs of each layer given inputs in X\n        \n        Parameters\n        ----------\n        X : input samples, standardized\n\n        Returns\n        -------\n        Outputs of all layers as list\n        """\n        # unpack self.all_weights to self.Ws\n        i=0\n        for layerI in range(len(self.shapes)):\n            shapeX=self.shapes[layerI][0]\n            shapeY=self.shapes[layerI][1]\n            self.Ws[layerI]=self.all_weights[i:i+shapeX*shapeY].reshape(shapeX,shapeY)\n            i+=shapeX*shapeY\n        \n        self.Ys=[]\n        for layerI in range(len(self.n_hidden_units_by_layers)):\n            X=np.tanh(self.addOnes(X) @ self.Ws[layerI])\n            self.Ys.append(X)\n        X=self.addOnes(X)@self.Ws[-1]\n        self.Ys.append(X)\n        # Append output of each layer to list in self.Ys, then return it.\n        # ...\n\n    # Function to be minimized by optimizer method, mean squared error\n    def error_f(self, X, T):\n        """Calculate output of net and its mean squared error \n\n        Parameters\n        ----------\n        X : two-dimensional numpy array\n            number of samples  x  number of input components\n        T : two-dimensional numpy array\n            number of samples  x  number of output components\n\n        Returns\n        -------\n        Mean square error as scalar float that is the mean\n        square error over all samples\n        """\n        self._forward(X)\n        error = (T - self.Ys[-1]) * self.T_stds \n        self.error_trace.append(error)\n        summation = 0  #variable to store the summation of differences\n        n = len(error) #finding total number of items in list\n        for i in range (n):  #looping through each element of the list\n            difference = error[i]**2\n            summation +=difference  \n        MSE = summation/n  #dividing summation by total values to obtain average\n        \n \n# # #       errors.append(nnet.get_error_trace())\n \n# # #       plt.plot(self.all_weights, \'*-\', label=\'w\')\n\n# # #       plt.show()\n\n\n#         plt.plot(self.Ys[-1], \'o-\', label=\'Model \')\n#         plt.plot(T, \'*-\', label=\'Train\')\n\n#         plt.draw()\n#         plt.pause(0.00001)\n#         plt.clf()\n \n \n        # Call _forward, calculate mean square error and return it.\n        # ...\n        return MSE\n        \n        # Call _forward, calculate mean square error and return it.\n        # ...\n\n    # Gradient of function to be minimized for use by optimizer method\n    def gradient_f(self, X, T):\n        """Returns gradient wrt all weights. Assumes _forward already called.\n\n        Parameters\n        ----------\n        X : two-dimensional numpy array\n            number of samples  x  number of input components\n        T : two-dimensional numpy array\n            number of samples  x  number of output components\n\n        Returns\n        -------\n        Vector of gradients of mean square error wrt all weights\n        """\n \n        self._forward( X)\n        # Assumes forward_pass just called with layer outputs saved in self.Ys.\n        n_samples = X.shape[0]\n        n_outputs = T.shape[1]\n        n_layers = len(self.n_hidden_units_by_layers) + 1\n\n        # D is delta matrix to be back propagated\n        D = (T - self.Ys[-1])  \n        self.Grads =  [None] *n_layers\n\n        # Step backwards through the layers to back-propagate the error (D)\n        for layeri in range(n_layers - 1, -1, -1):\n            # gradient of all but bias weights\n \n            # Back-propagate this layer\'s delta to previous layer\n            if layeri > 0:\n                self.Grads[layeri]= -self.addOnes(self.Ys[layeri-1]).T@D\n                D =D@self.Ws[layeri][1:,:].T*(1-self.Ys[layeri-1]**2)  \n            else:\n                self.Grads[layeri]= -self.addOnes(X).T@D\n         \n        \n        self.all_gradients=np.empty([0,1])\n        \n        for layerI in range(n_layers):\n             \n#           haha=np.vstack((self.all_weights,wI.reshape(-1,1)))\n            self.all_gradients=np.vstack((self.all_gradients,self.Grads[layerI].reshape(-1,1))) \n            \n            \n            \n            \n        return self.all_gradients\n\n    def use(self, X):\n        """Return the output of the network for input samples as rows in X\n\n        Parameters\n        ----------\n        X : two-dimensional numpy array\n            number of samples  x  number of input components, unstandardized\n\n        Returns\n        -------\n        Output of neural network, unstandardized, as numpy array\n        of shape  number of samples  x  number of outputs\n        """\n \n        X=(X-self.X_means)/self.X_stds\n        self._forward( X)\n        Y=self.Ys[-1]\n        Y=Y*self.T_stds+self.T_means\n        return Y \n \n    def get_error_trace(self):\n        """Returns list of standardized mean square error for each epoch"""\n        return self.error_trace')


# ## Example Results

# Here we test the `NeuralNetwork` class with some simple data.  
# 

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import neuralnetwork as nn

#%% main functions
 
X = np.arange(-2, 2, 0.05).reshape(-1, 1)
T = np.sin(X) * np.sin(X * 10)

errors = []
n_epochs = 1000
method_rhos = [  
    
#                 ('adam', 0.01),
                ('scg', None)]
 
for method, rho in method_rhos:
    nnet = nn.NeuralNetwork(X.shape[1], [30, 30], 1)
    nnet.train(X, T, n_epochs, method=method, learning_rate=rho)
    Y = nnet.use(X)
    plt.plot(X, Y, 'o-', label='Model ' + method)
    plt.plot(X, T, 'o', label='Train')
    errors.append(nnet.get_error_trace())
    plt.show()
    exit()


plt.plot(X, T, 'o', label='Train')
plt.xlabel('X')
plt.ylabel('T or Y')
plt.legend();


# In[ ]:


plt.figure(2)
plt.clf()
for error_trace in errors:
    plt.plot(error_trace)
plt.xlabel('Epoch')
plt.ylabel('Standardized error')
plt.legend([mr[0] for mr in method_rhos]);


# Your results will not be the same, but your code should complete and make plots somewhat similar to these.

# ## Application to Boston Housing Data

# Download data from [Boston House Data at Kaggle](https://www.kaggle.com/fedesoriano/the-boston-houseprice-data). Read it into python using the `pandas.read_csv` function.  Assign the first 13 columns as inputs to `X` and the final column as target values to `T`.  Make sure `T` is two-dimensional.

# Before training your neural networks, partition the data into training and testing partitions, as shown here.

# In[ ]:


def partition(X, T, train_fraction):
    n_samples = X.shape[0]
    rows = np.arange(n_samples)
    np.random.shuffle(rows)
    
    n_train = round(n_amples * train_fraction)
    
    Xtrain = X[rows[:ntrain], :]
    Ttrain = T[rows[:ntrain], :]
    Xtest = X[rows[ntrain:], :]
    Ttest = T[rows[ntrain:], :]
    
def rmse(T, Y):
    return np.sqrt(np.mean((T - Y)**2))


# In[ ]:


# Assuming you have assigned `X` and `T` correctly.

Xtrain, Train, Xtest, Ttest = partition(X, T, 0.8)  


# Write and run code using your `NeuralNetwork` class to model the Boston housing data. Experiment with all three optimization methods and a variety of neural network structures (numbers of hidden layer and units), learning rates, and numbers of epochs. Show results for at least three different network structures, learning rates, and numbers of epochs for each method.  Show your results using print statements that include the method, network structure, number of epochs, learning rate, and RMSE on training data and RMSE on testing data.
# 
# Try to find good values for the RMSE on testing data.  Discuss your results, including how good you think the RMSE values are by considering the range of house values given in the data. 

# # Grading
# 
# <font color='red'>A2grader.tar is coming soon.</font>
# 
# Your notebook will be run and graded automatically. Test this grading process by first downloading [A2grader.tar](http://www.cs.colostate.edu/~anderson/cs545/notebooks/A2grader.tar) and extract `A2grader.py` from it. Run the code in the following cell to demonstrate an example grading session.  The remaining 20 points will be based on your discussion of this assignment.
# 
# A different, but similar, grading script will be used to grade your checked-in notebook. It will include additional tests. You should design and perform additional tests on all of your functions to be sure they run correctly before checking in your notebook.  
# 
# For the grading script to run correctly, you must first name this notebook as 'Lastname-A2.ipynb' with 'Lastname' being your last name, and then save this notebook.

# In[ ]:


get_ipython().run_line_magic('run', '-i A2grader.py')


# # Check-In
# 
# Do not include this section in your notebook.
# 
# Name your notebook ```Lastname-A2.ipynb```.  So, for me it would be ```Anderson-A2.ipynb```.  Submit the file using the ```Assignment 2``` link on [Canvas](https://colostate.instructure.com/courses/131494).

# # Extra Credit
# 
# Apply your multilayer neural network code to a regression problem using data that you choose 
# from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets.php). Pick a dataset that
# is listed as being appropriate for regression.
