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


from ipywidgets import interact
maxSamples = 100
nSets = 10000
values = np.random.uniform(0,1,(maxSamples,nSets))
plt.figure()

# @interact(nSamples=(1,maxSamples))
nSamples=1

 
sums = np.sum(values[:nSamples,:],axis=0)
aa=values[:nSamples,:]
# plt.clf()
# plt.hist(sums, 20, facecolor='green')
a=[1,2,3,4,5]
b=a[:4]
print(b)