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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
import optimizers as opt




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv('parkinsons.data')
print(data.shape)
print(data.columns)
T = data['status'].values
T = T.reshape((-1, 1))
Tname = 'status'
X = data
X = X.drop(['status', 'name'], axis=1)
Xnames = X.columns.tolist()
X = X.values
X.shape, Xnames
print(f'{" ":20s} {"mean":9s} {"stdev":9s}')
for i in range(len(Xnames)):
	print(f'{Xnames[i]:20s} {np.mean(X[:, i]):9.3g} {np.std(X[:, i]):9.3g}')

uniq = np.unique(T)
print('   Value  Occurrences')
for i in uniq:
	print(f'{i:7.1g} {np.sum(T==i):10d}')
	
trainf = 0.8
healthyI,_ = np.where(T == 0)
parkI,_ = np.where(T == 1)
healthyI = np.random.permutation(healthyI)
parkI = np.random.permutation(parkI)

nHealthy = round(trainf * len(healthyI))
nPark = round(trainf * len(parkI))
rowsTrain = np.hstack((healthyI[:nHealthy], parkI[:nPark]))
Xtrain = X[rowsTrain, :]
Ttrain = T[rowsTrain, :]
rowsTest = np.hstack((healthyI[nHealthy:], parkI[nPark:]))
Xtest =  X[rowsTest, :]
Ttest =  T[rowsTest, :]


print('Xtrain is {:d} by {:d}. Ttrain is {:d} by {:d}'.format(*(Xtrain.shape + Ttrain.shape)))
uniq = np.unique(Ttrain)
print('   Value  Occurrences')
for i in uniq:
    print(f'{i:7.1g} {np.sum(Ttrain == i):10d}')

    
print('Xtest is {:d} by {:d}. Ttest is {:d} by {:d}'.format(*(Xtest.shape + Ttest.shape)))
uniq = np.unique(Ttest)
print('   Value  Occurrences')
for i in uniq:
    print(f'{i:7.1g} {np.sum(Ttest == i):10d}')
	
# =============================================================================
#Least Squares Solution (Linear Fitting)
# =============================================================================
def train(X, T, lamb=0):
    means = X.mean(0)
    stds = X.std(0)
    n,d = X.shape
    Xs = (X - means) / stds
    Xs1 = np.insert(Xs , 0, 1, axis=1)
    lambDiag = np.eye(d + 1) * lamb
    lambDiag[0, 0] = 0
    w = np.linalg.lstsq( Xs1.T @ Xs1 + lambDiag, Xs1.T @ T, rcond=None)[0]
    return {'w': w, 'means':means, 'stds':stds}

def use(model, X):
    Xs = (X - model['means']) / model['stds']
    Xs1 = np.insert(Xs , 0, 1, axis=1)
    return Xs1 @ model['w']
	
model = train(Xtrain, Ttrain)

Xnames.insert(0,'bias')
for i in range(len(Xnames)):
    print('{:2d} {:>20s} {:10.3g}'.format(i, Xnames[i], model['w'][i][0]))

def convertTo01(Y):
    distFromTarget = np.abs(Y - [0,1])
    whichTargetClosest = np.argmin(distFromTarget, axis=1).reshape((-1, 1))
    return whichTargetClosest  # column index equivalent to 0 and 1 targets
convertTo01(np.array([0.1, 1.1, -0.5, 0.56]).reshape((-1,1)))

Ytrain = use(model, Xtrain)

predictedTrain = convertTo01(Ytrain)

percentCorrectTrain = np.sum(predictedTrain == Ttrain) / Ttrain.shape[0] * 100.0

Ytest = use(model, Xtest)

predictedTest = convertTo01(Ytest)
percentCorrectTest = np.sum(predictedTest == Ttest) / float(Ttest.shape[0]) * 100.0

print('Percent Correct: Training {:6.1f} Testing {:6.1f}'.format(percentCorrectTrain, percentCorrectTest))

plt.figure()
plt.subplot(2, 1 ,1)
plt.plot(np.hstack((Ttrain, predictedTrain)), 'o-', alpha=0.5)
plt.ylim(-0.1, 1.1) # so markers will show
plt.xlabel('Sample Index')
plt.ylabel('Class')
plt.title('Training Data')
plt.legend(('Actual', 'Predicted'), loc='center')

plt.subplot(2, 1, 2)
plt.plot(np.hstack((Ttest, predictedTest)), 'o-', alpha=0.5)
plt.ylim(-0.1, 1.1)
plt.xlabel('Sample Index')
plt.ylabel('Class')
plt.title('Testing Data')
plt.legend(('Actual', 'Predicted'), loc='center');

plt.tight_layout()
#%%
plt.figure()

plt.subplot(2, 1, 1)
plt.plot(np.hstack((Ttrain, predictedTrain, Ytrain)),'o-', alpha=0.5)
plt.ylim(-0.1, 1.1) # so markers will show
plt.xlabel('Sample Index')
plt.ylabel('Class')
plt.title('Training Data')
plt.legend(('Actual', 'Predicted', 'Cont. Val.'), loc='center')

plt.subplot(2, 1, 2)
plt.plot(np.hstack((Ttest, predictedTest, Ytest)), 'o-', alpha=0.5)
plt.ylim(-0.1, 1.1)
plt.xlabel('Sample Index')
plt.ylabel('Class')
plt.title('Testing Data')
plt.legend(('Actual', 'Predicted', 'Cont. Val.'), loc='center')

plt.tight_layout()


counts = np.array([[2, 6, 4], [3, 1, 2]])
counts
jarNames = ['red', 'blue']
fruitNames = ['apple', 'orange', 'strawberry']
def printTable(label, data):
    print
    print(label)
    print('   {:>9s} {:>7s} {:>9s}'.format(*fruitNames))
    for i in [0, 1]:
        d = data[i, :].tolist()
        print('{:4s} {:7.3g} {:7.3g} {:7.3g} {:7.3g}'.format(*([jarNames[i]] + d + [sum(d)])))
    colTotals = np.sum(data, axis=0).tolist()
    print('     {:7.3g} {:7.3g} {:7.3g} {:7.3g}'.format(*(colTotals + [sum(colTotals)])))

printTable('counts', counts)
jarSums = np.sum(counts, axis=1).reshape((2, 1))
jarSums
pFruitGivenJar = counts / jarSums
printTable('Prob(Fruit|Jar)', pFruitGivenJar)
pJar = np.array([[0.6], [0.4]])
pJar

pFruitAndJar = pFruitGivenJar * pJar
printTable('Prob(Fruit,Jar)', pFruitAndJar)
pFruit = np.sum(pFruitAndJar, axis=0)
pFruit
pJarGivenFruit = pFruitAndJar / pFruit
printTable('Prob(Jar|Fruit)', pJarGivenFruit)

#%%
plt.figure()
xs = np.linspace(-5,10,1000)
mu = 5.5
plt.plot(xs, 1/np.sqrt((xs-mu)**2))
plt.ylim(0,20)
plt.plot([mu, mu], [0, 20], 'r--',lw=2)
plt.xlabel('$x$')
plt.ylabel('$p(x)$');

plt.figure()
plt.plot(xs, 1/2**np.sqrt((xs-mu)**2))
plt.plot([mu, mu], [0, 1], 'r--',lw=3)
plt.xlabel('$x$')
plt.ylabel('$p(x)$');

plt.figure()
plt.plot(xs, 1/2**(xs-mu)**2)
plt.plot([mu, mu], [0, 1], 'r--',lw=3)
plt.xlabel('$x$')
plt.ylabel('$p(x)$');

plt.figure()
plt.plot(xs, 1/2**(0.1 * (xs-mu)**2))
plt.plot([mu, mu], [0, 1], 'r--',lw=3)
plt.xlabel('$x$')
plt.ylabel('$p(x)$');

plt.figure()
plt.plot(xs, np.exp(-0.1 * (xs-mu)**2))
plt.plot([mu, mu], [0, 1], 'r--',lw=3)
plt.xlabel('$x$')
plt.ylabel('$p(x)$');

import ipywidgets as widgets

# set up plot
fig, ax = plt.subplots(figsize=(6, 4))
ax.set_ylim([-4, 4])
ax.grid(True)
 
# generate x values
x = np.linspace(0, 2 * np.pi, 100)
 
 
def my_sine(x, w, amp, phi):
    """
    Return a sine for x with angular frequeny w and amplitude amp.
    """
    return amp*np.sin(w * (x-phi))
 
 
@widgets.interact(w=(0, 10, 1), amp=(0, 4, .1), phi=(0, 2*np.pi+0.01, 0.01))
def update(w = 1.0, amp=1, phi=0):
    """Remove old lines from plot and plot new one"""
    [l.remove() for l in ax.lines]
    ax.plot(x, my_sine(x, w, amp, phi), color='C0')

from ipywidgets import interact
maxSamples = 100
nSets = 10000
values = np.random.uniform(0,1,(maxSamples,nSets))
plt.figure()

@interact(nSamples=(1,maxSamples))
def sumOfN(nSamples=1):
    sums = np.sum(values[:nSamples,:],axis=0)
    plt.clf()
    plt.hist(sums, 20, facecolor='green')

def normald(X, mu, sigma):
    """ normald:
       X contains samples, one per row, N x D. 
       mu is mean vector, D x 1.
       sigma is covariance matrix, D x D.  """
    D = X.shape[1]
    detSigma = sigma if D == 1 else np.linalg.det(sigma)
    if detSigma == 0:
        raise np.linalg.LinAlgError('normald(): Singular covariance matrix')
    sigmaI = 1.0/sigma if D == 1 else np.linalg.inv(sigma)
    normConstant = 1.0 / np.sqrt((2*np.pi)**D * detSigma)
    diffv = X - mu.T # change column vector mu to be row vector
    return normConstant * np.exp(-0.5 * np.sum(np.dot(diffv, sigmaI) * diffv, axis=1))[:,np.newaxis]

X = np.array([[1,2],[3,5],[2.1,1.9]])
mu = np.array([[2],[2]])
Sigma = np.array([[1,0],[0,1]])
print(X)
print(mu)
print(Sigma)
normald(X, mu, Sigma)

x = np.linspace(-5, 5, 50)
y = x.copy()
xmesh, ymesh = np.meshgrid(x, y)
xmesh.shape, ymesh.shape

X = np.vstack((xmesh.flat, ymesh.flat)).T
X.shape

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

fig = plt.figure()
ax = plt.subplot(projection='3d')
# ax.set_aspect("equal")

mu = np.array([[2,-2]]).T
Sigma = np.array([[1,0],[0,1]])

Z = normald(X, mu, Sigma)
Zmesh = Z.reshape(xmesh.shape)
surface = ax.plot_surface(xmesh, ymesh, Zmesh, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False);

plt.colorbar(surface, shrink=0.3);

# =============================================================================
# QDA: Quadratic Discriminant Analysis
# =============================================================================

D = 1  # number of components in each sample
N = 10  # number of samples in each class
X1 = np.random.normal(0.0, 1.0, (N, D))
T1 = np.array([1]*N).reshape((N, 1))
X2 = np.random.normal(4.0, 1.5, (N, D))  # wider variance
T2 = np.array([2]*N).reshape((N, 1))

data = np.hstack(( np.vstack((X1, X2)), np.vstack((T1, T2))))
data.shape

X = data[:, 0:D]
T = data[:, -1:]
means = np.mean(X, 0)
stds = np.std(X, 0)
Xs = (X - means) / stds

Sigma = np.array([[1, 2], [2, 1]])
Sigma @ np.linalg.inv(Sigma)
Sigma @ np.linalg.pinv(Sigma)
Sigma = np.array([[1, 2], [1, 2]])
Sigma

def discQDA(X, means, stds, mu, Sigma, prior):
    Xc = (X - means) / stds - mu
    if Sigma.size == 1:
        Sigma = np.asarray(Sigma).reshape((1,1))
    det = np.linalg.det(Sigma)        
    # if det == 0:
    #   raise np.linalg.LinAlgError('discQDA(): Singular covariance matrix')
    SigmaInv = np.linalg.pinv(Sigma)     # pinv in case Sigma is singular
    return -0.5 * np.log(det) \
           - 0.5 * np.sum(np.dot(Xc, SigmaInv) * Xc, axis=1).reshape((-1,1)) \
           + np.log(prior)

class1rows = (T==1).reshape((-1))
class2rows = (T==2).reshape((-1))

mu1 = np.mean(Xs[class1rows, :], axis=0)
mu2 = np.mean(Xs[class2rows, :], axis=0)

Sigma1 = np.cov(Xs[class1rows, :].T)
Sigma2 = np.cov(Xs[class2rows, :].T)

N1 = np.sum(class1rows)
N2 = np.sum(class2rows)
N = len(T)
prior1 = N1 / float(N)
prior2 = N2 / float(N)

nNew = 100
newData = np.linspace(-5.0, 10.0, nNew).repeat(D).reshape((nNew, D))

d1 = discQDA(newData, means, stds, mu1, Sigma1, prior1)
d2 = discQDA(newData, means, stds, mu2, Sigma2, prior2)

def normald(X, mu, sigma):
    """ normald:
       X contains samples, one per row, N x D. 
       mu is mean vector, D x 1.
       sigma is covariance matrix, D x D.  """
    D = X.shape[1]
    detSigma = sigma if D == 1 else np.linalg.det(sigma)
    if detSigma == 0:
        raise np.linalg.LinAlgError('normald(): Singular covariance matrix')
    sigmaI = 1.0/sigma if D == 1 else np.linalg.inv(sigma)
    normConstant = 1.0 / np.sqrt((2*np.pi)**D * detSigma)
    diffv = X - mu.T # change column vector mu to be row vector
    return normConstant * np.exp(-0.5 * np.sum(np.dot(diffv, sigmaI) * diffv, axis=1))[:,np.newaxis]

plt.figure()
plt.subplot(3, 1, 1)
plt.plot(newData[:, 0],np.hstack((d1, d2)))
plt.ylabel("QDA Discriminant Functions")

# Plot generative distributions  p(x | Class=k)  starting with discriminant functions
plt.subplot(3, 1, 2)

probs = np.exp( np.hstack((d1, d2)) - 0.5  *D * np.log(2 * np.pi)  - np.log(np.array([[prior1, prior2]])))

plt.plot(newData[:,0], probs)
plt.ylabel("QDA P(x|Class=k)\n from disc funcs", multialignment="center")

# Plot generative distributions  p(x | Class=k)  using normald    ERROR HERE
plt.subplot(3, 1 ,3)
newDataS = (newData - means) / stds

probs = np.hstack((normald(newDataS, mu1, Sigma1),
                   normald(newDataS, mu2, Sigma2)))
plt.plot(newData, probs)
plt.ylabel("QDA P(x|Class=k)\n using normald", multialignment="center");

D = 20  # number of components in each sample
N = 10  # number of samples in each class
X1 = np.random.normal(0.0, 1.2, (N, D))
T1 = np.array([1]*N).reshape((N, 1))
X2 = np.random.normal(4.0, 1.8, (N, D))  # wider variance
T2 = np.array([2]*N).reshape((N, 1))

data = np.hstack(( np.vstack((X1, X2)), np.vstack((T1, T2))))
X = data[:, 0:D]
T = data[:, -1]
means, stds = np.mean(X,0), np.std(X,0)
Xs = (X-means)/stds

class1rows = T==1
class2rows = T==2

mu1 = np.mean(Xs[class1rows,:],axis=0)
mu2 = np.mean(Xs[class2rows,:],axis=0)

Sigma1 = np.cov(Xs[class1rows,:].T)
Sigma2 = np.cov(Xs[class2rows,:].T)

N1 = np.sum(class1rows)
N2 = np.sum(class2rows)
N = len(T)
prior1 = N1 / float(N)
prior2 = N2 / float(N)

nNew = 100
newData = np.linspace(-5.0,10.0,nNew).repeat(D).reshape((nNew,D))

d1 = discQDA(newData,means,stds,mu1,Sigma1,prior1)
d2 = discQDA(newData,means,stds,mu2,Sigma2,prior2)

plt.figure()
plt.subplot(3,1,1)
plt.plot(newData[:,0],np.hstack((d1,d2)))
plt.ylabel("QDA Discriminant Functions")
# Plot generative distributions  p(x | Class=k)  starting with discriminant functions
plt.subplot(3,1,2)
probs = np.exp( np.hstack((d1,d2)) - 0.5*D*np.log(2*np.pi) - np.log(np.array([[prior1,prior2]])))
plt.plot(newData[:,0],probs)
plt.ylabel("QDA P(x|Class=k)\n from disc funcs", multialignment="center")

# Plot generative distributions  p(x | Class=k)  using normald
plt.subplot(3,1,3)
newDataS = (newData-means)/stds
probs = np.hstack((normald(newDataS,mu1,Sigma1),
                   normald(newDataS,mu2,Sigma2)))
plt.plot(newData[:,0],probs)
plt.ylabel("QDA P(x|Class=k)\n using normald", multialignment="center");

# Fit generative models (Normal distributions) to each class
means,stds = np.mean(Xtrain, 0), np.std(Xtrain, 0)
Xtrains = (Xtrain - means) / stds

Ttr = (Ttrain==0).reshape((-1))
mu1 = np.mean(Xtrains[Ttr, :], axis=0)
cov1 = np.cov(Xtrains[Ttr, :].T)
Ttr = (Ttrain.ravel()==1).reshape((-1))
mu2 = np.mean(Xtrains[Ttr, :],axis=0)
cov2 = np.cov(Xtrains[Ttr, :].T)

d1 = discQDA(Xtrain, means, stds, mu1, cov1, float(nHealthy)/(nHealthy+nPark))
d2 = discQDA(Xtrain, means, stds, mu2, cov2, float(nPark)/(nHealthy+nPark))
predictedTrain = np.argmax(np.hstack((d1, d2)), axis=1)

d1t = discQDA(Xtest, means, stds, mu1, cov1, float(nHealthy)/(nHealthy+nPark))
d2t = discQDA(Xtest, means, stds, mu2, cov2, float(nPark)/(nHealthy+nPark))
predictedTest = np.argmax(np.hstack((d1t, d2t)), axis=1)

def percentCorrect(p, t):
    return np.sum(p.ravel()==t.ravel()) / float(len(t)) * 100

print('Percent correct: Train', percentCorrect(predictedTrain,Ttrain), 'Test', percentCorrect(predictedTest,Ttest))

def runPark(filename, trainFraction):
    f = open(filename,"r")
    header = f.readline()
    names = header.strip().split(',')[1:]

    data = np.loadtxt(f ,delimiter=',', usecols=1+np.arange(23))

    targetColumn = names.index("status")
    XColumns = np.arange(23)
    XColumns = np.delete(XColumns, targetColumn)
    X = data[:, XColumns]
    T = data[:, targetColumn].reshape((-1,1)) # to keep 2-d matrix form
    names.remove("status")

    healthyI,_ = np.where(T == 0)
    parkI,_ = np.where(T == 1)
    healthyI = np.random.permutation(healthyI)
    parkI = np.random.permutation(parkI)

    nHealthy = round(trainFraction*len(healthyI))
    nPark = round(trainf*len(parkI))
    rowsTrain = np.hstack((healthyI[:nHealthy], parkI[:nPark]))
    Xtrain = X[rowsTrain, :]
    Ttrain = T[rowsTrain, :]
    rowsTest = np.hstack((healthyI[nHealthy:], parkI[nPark:]))
    Xtest =  X[rowsTest, :]
    Ttest =  T[rowsTest, :]

    means, stds = np.mean(Xtrain, 0), np.std(Xtrain, 0)
    Xtrains = (Xtrain-means)/stds

    Ttr = (Ttrain==0).reshape((-1))
    mu1 = np.mean(Xtrains[Ttr, :], axis=0)
    cov1 = np.cov(Xtrains[Ttr, :].T)
    Ttr = (Ttrain.ravel()==1).reshape((-1))
    mu2 = np.mean(Xtrains[Ttr, :],axis=0)
    cov2 = np.cov(Xtrains[Ttr, :].T)

    d1 = discQDA(Xtrain, means, stds, mu1, cov1, float(nHealthy)/(nHealthy+nPark))
    d2 = discQDA(Xtrain, means, stds, mu2, cov2, float(nPark)/(nHealthy+nPark))
    predictedTrain = np.argmax(np.hstack((d1, d2)), axis=1)

    d1t = discQDA(Xtest, means, stds, mu1, cov1, float(nHealthy)/(nHealthy+nPark))
    d2t = discQDA(Xtest, means, stds, mu2, cov2, float(nPark)/(nHealthy+nPark))
    predictedTest = np.argmax(np.hstack((d1t, d2t)), axis=1)

    print('Percent correct: Train', percentCorrect(predictedTrain, Ttrain), 'Test', percentCorrect(predictedTest,Ttest))

def percentCorrect(p, t):
    return np.sum(p.ravel()==t.ravel()) / float(len(t)) * 100

runPark('parkinsons.data', 0.8)
runPark('parkinsons.data', 0.8)
runPark('parkinsons.data', 0.8)
runPark('parkinsons.data', 0.8)


# =============================================================================
# LDA: Linear Discriminant Analysis
# =============================================================================

def discLDA(X, means,stds, mu, Sigma, prior):
    X = (X-means)/stds
    if Sigma.size == 1:
        Sigma = np.asarray(Sigma).reshape((1,1))
    det = np.linalg.det(Sigma)        
    # if det == 0:
    #    raise np.linalg.LinAlgError('discQDA(): Singular covariance matrix')
    SigmaInv = np.linalg.pinv(Sigma)     # pinv in case Sigma is singular
    mu = mu.reshape((-1,1)) # make mu a column vector
    # pdb.set_trace()
    return np.dot(np.dot(X,SigmaInv), mu) - 0.5 * np.dot(np.dot(mu.T,SigmaInv), mu) + np.log(prior)

def runPark(filename, trainFraction):
    f = open(filename,"r")
    header = f.readline()
    names = header.strip().split(',')[1:]

    data = np.loadtxt(f ,delimiter=',', usecols=1+np.arange(23))

    targetColumn = names.index("status")
    XColumns = np.arange(23)
    XColumns = np.delete(XColumns, targetColumn)
    X = data[:, XColumns]
    T = data[:, targetColumn].reshape((-1,1)) # to keep 2-d matrix form
    names.remove("status")

    healthyI,_ = np.where(T == 0)
    parkI,_ = np.where(T == 1)
    healthyI = np.random.permutation(healthyI)
    parkI = np.random.permutation(parkI)

    nHealthy = round(trainFraction*len(healthyI))
    nPark = round(trainf*len(parkI))
    rowsTrain = np.hstack((healthyI[:nHealthy], parkI[:nPark]))
    Xtrain = X[rowsTrain, :]
    Ttrain = T[rowsTrain, :]
    rowsTest = np.hstack((healthyI[nHealthy:], parkI[nPark:]))
    Xtest =  X[rowsTest, :]
    Ttest =  T[rowsTest, :]

    means,stds = np.mean(Xtrain,0), np.std(Xtrain,0)
    Xtrains = (Xtrain-means)/stds

    Ttr = (Ttrain==0).reshape((-1))
    mu1 = np.mean(Xtrains[Ttr, :],axis=0)
    cov1 = np.cov(Xtrains[Ttr, :].T)
    Ttr = (Ttrain.ravel()==1).reshape((-1))
    mu2 = np.mean(Xtrains[Ttr, :],axis=0)
    cov2 = np.cov(Xtrains[Ttr, :].T)

    d1 = discQDA(Xtrain, means, stds, mu1, cov1, float(nHealthy)/(nHealthy+nPark))
    d2 = discQDA(Xtrain, means, stds, mu2, cov2, float(nPark)/(nHealthy+nPark))
    predictedTrain = np.argmax(np.hstack((d1, d2)),axis=1)

    d1t = discQDA(Xtest, means, stds, mu1, cov1, float(nHealthy)/(nHealthy+nPark))
    d2t = discQDA(Xtest, means, stds, mu2, cov2, float(nPark)/(nHealthy+nPark))
    predictedTest = np.argmax(np.hstack((d1t, d2t)), axis=1)

    print('QDA Percent correct: Train', percentCorrect(predictedTrain, Ttrain), 'Test', percentCorrect(predictedTest,Ttest))

    covMean = (cov1 * nHealthy + cov2 * nPark) / (nHealthy+nPark)
    d1 = discLDA(Xtrain, means, stds, mu1, covMean, float(nHealthy)/(nHealthy+nPark))
    d2 = discLDA(Xtrain, means, stds, mu2, covMean, float(nPark)/(nHealthy+nPark))
    predictedTrain = np.argmax(np.hstack((d1, d2)), axis=1)

    d1t = discLDA(Xtest, means, stds, mu1, covMean, float(nHealthy)/(nHealthy+nPark))
    d2t = discLDA(Xtest, means, stds, mu2, covMean, float(nPark)/(nHealthy+nPark))
    predictedTest = np.argmax(np.hstack((d1t, d2t)), axis=1)
    print('LDA Percent correct: Train', percentCorrect(predictedTrain, Ttrain), 'Test', percentCorrect(predictedTest,Ttest))

def percentCorrect(p, t):
    return np.sum(p.ravel()==t.ravel()) / float(len(t)) * 100

runPark('parkinsons.data', 0.8)

for i in range(5):
    runPark('parkinsons.data', 0.8)
    print()

import sys
sys.float_info.epsilon, np.log(sys.float_info.epsilon)

