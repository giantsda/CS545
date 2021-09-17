#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd  # for display and clear_output
import os
import copy
import signal
import os
import numpy as np


# %matplotlib inline

def addOnes(A):
	return np.insert(A, 0, 1, axis=1)

def rmse(T, Y, Tstds):
	error = (T - Y) * Tstds 
	return np.sqrt(np.mean(error ** 2))

def forward(X, U, V, W):
	Zu= np.tanh(X @ U)
	Zv=np.tanh(addOnes(Zu) @ V)
	Z1 = addOnes(Zv)
	Y= Z1 @ W
	return Zu, Zv, Y

def gradient(X, T, Zu, Zv, Y, U, V, W):
	
	
	Dw = T - Y
	grad_wrt_W=-addOnes(Zv).T@Dw
	Dv=Dw@W[1:,:].T*(1-Zv**2)
	grad_wrt_V=-addOnes(Zu).T@Dv
	Du=Dv@V[1:,:].T*(1-Zu**2)
	grad_wrt_U=-X.T@Du

# 	grad_wrt_W=-addOnes(Zv).T@(T-Y)
# 	grad_wrt_V=-addOnes(Zu).T@(((T-Y)@W[1:, :].T)*(1-Zv**2))
# 	grad_wrt_U=-X.T@(((T-Y)@W[1:, :].T)*(1-Zv**2)@V[1:, :].T*(1-Zu**2))
 
    
	return grad_wrt_U, grad_wrt_V, grad_wrt_W

def use(X, X_means, X_stds, T_means, T_stds, U, V, W):
	X=(X-X_means)/X_stds
	Zu, Zv, Y=forward(addOnes(X), U, V, W)
	Y=Y*T_stds+T_means
	return Y 
 
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
		 
		grad_wrt_U, grad_wrt_V, grad_wrt_W=gradient(XtrainS1, TtrainS, Zu, Zv, Y, U, V, W)

		# Take step down the gradient
		U = U - rhoI * grad_wrt_U
		V = V - rhoI * grad_wrt_V  
		W = W - rhoI * grad_wrt_W  		
	 
		error.append(rmse(TtrainS, Y, Tstds))
		if epoch==95:
			aaa=111

# 			exit();
			
		if epoch%50==0:
			plt.subplot(221)
			
			
			plt.plot(error)
			plt.subplot(222)
			plt.plot(Y)
			plt.plot(TtrainS)
			plt.draw()
			plt.pause(0.00001)
			plt.clf()

# 			ipd.display(fig)
	return U, V, W, Xmeans, Xstds, Tmeans, Tstds








X = np.arange(4).reshape(-1, 1)
T = np.log(X + 10)

ni = 1
nu = 2
nv = 3
U = np.arange((ni + 1) * nu).reshape(ni + 1, nu) * 0.1
V = (np.arange((nu + 1) * nv).reshape(nu + 1, nv) - 6) * 0.1
W = np.arange(nv + 1).reshape(nv + 1, 1) * -0.1

Zu = np.array([[0.        , 0.09966799],
			   [0.19737532, 0.37994896],
			   [0.37994896, 0.60436778],
			   [0.53704957, 0.76159416]])
Zv = np.array([[-0.53704957, -0.45424278, -0.36276513],
			   [-0.57783916, -0.46328044, -0.3308191 ],
			   [-0.61316945, -0.47426053, -0.30690171],
			   [-0.64173317, -0.4863364 , -0.29258059]])
Y = np.array([[0.25338305],
			  [0.24968573],
			  [0.24823956],
			  [0.24921478]])

grad_wrt_U, grad_wrt_V, grad_wrt_W = gradient(addOnes(X), T, Zu, Zv, Y, U, V, W)

grad_wrt_U_answer = np.array([[-0.5952239 ,  0.43237751],
							  [-0.82940893,  0.53004459]])

grad_wrt_V_answer = np.array([[0.56468907, 1.36302356, 2.35084051],
					  [0.15442332, 0.3882369 , 0.68537352],
					  [0.25698882, 0.63947139, 1.12241063]])
grad_wrt_W_answer = np.array([[-8.74981325],
					  [ 5.19938229],
					  [ 4.11304763],
					  [ 2.81802374]])
	
	
	
	
	
	
	
	
	
	
	

n_inputs = 3
n_hiddens = [10, 20]
n_outputs = 2
n_samples = 5

X = np.arange(n_samples * n_inputs).reshape(n_samples, n_inputs) * 0.1
X_means = np.mean(X, axis=0)
X_stds = np.std(X, axis=0)
T_means = np.zeros((n_samples, n_outputs))
T_stds = np.ones((n_samples, n_outputs))



shapes=[]
layer_n=n_inputs
for layerI in range(2):
	layerI_N=n_hiddens[layerI]
	shapes.append([1 + layer_n, layerI_N])
	layer_n=layerI_N
shapes.append([1 + layer_n, n_outputs])
		
 
Ws= [1,1,1]
i=0
for layerI in range(len(shapes)):
	shapeX=shapes[layerI][0]
	shapeY=shapes[layerI][1]
	Ws[layerI]=0.1*np.ones((shapeX,shapeY))
	i+=shapeX*shapeY
		
Y = use(X, X_means, X_stds, T_means, T_stds, Ws[0], Ws[1], Ws[2])
print(Y)



 







Xtrain = np.arange(4).reshape(-1, 1)
Ttrain = Xtrain ** 2

Xtest = Xtrain + 0.5
Ttest = Xtest ** 2
 
U = np.array([[1, 2, 3], [4, 5, 6]])  # 2 x 3 matrix, for 2 inputs (include constant 1) and 3 units
V = np.array([[-1, 3], [1, 3], [-2, 1], [2, -4]]) # 2 x 3 matrix, for 3 inputs (include constant 1) and 2 units
W = np.array([[-1], [2], [3]])  # 3 x 1 matrix, for 3 inputs (include constant 1) and 1 ounit

X_means = np.mean(Xtrain, axis=0)
X_stds = np.std(Xtrain, axis=0)
Xtrain_st = (Xtrain - X_means) / X_stds

Zu, Zv, Y = forward(addOnes(Xtrain_st), U, V, W)
print('Zu = ', Zu)
print('Zv = ', Zv)
print('Y = ', Y)

T_means = np.mean(Ttrain, axis=0)
T_stds = np.std(Ttrain, axis=0)
Ttrain_st = (Ttrain - T_means) / T_stds
grad_wrt_U, grad_wrt_V, grad_wrt_W = gradient(Xtrain_st, Ttrain_st, Zu, Zv, Y, U, V, W)
print('grad_wrt_U = ', grad_wrt_U)
print('grad_wrt_V = ', grad_wrt_V)
print('grad_wrt_W = ', grad_wrt_W)

Y = use(Xtrain, X_means, X_stds, T_means, T_stds, U, V, W)
print(Y)

fig = plt.figure(figsize=(10, 20))

n = 30
Xtrain = np.linspace(0., 20.0, n).reshape((n, 1)) - 10
Ttrain = 0.2 + 0.05 * (Xtrain + 10) + 0.4 * np.sin(Xtrain + 10) + 0.2 * np.random.normal(size=(n, 1))

Xtest = Xtrain + 0.1 * np.random.normal(size=(n, 1))
Ttest = 0.2 + 0.05 * (Xtest + 10) + 0.4 * np.sin(Xtest + 10) + 0.2 * np.random.normal(size=(n, 1))

# U, V, W, X_means, X_stds, T_means, T_stds = train(Xtrain, Ttrain, 5, 5, 100, 0.01)
# Y = use(Xtrain, X_means, X_stds, T_means, T_stds, U, V, W)
# plt.plot(Xtrain, Ttrain)
# plt.plot(Xtrain, Y)
# plt.legend(('Ttrain', 'Y'), loc='upper left')
# plt.show()
 

U, V, W, X_means, X_stds, T_means, T_stds = train(Xtrain, Ttrain, 50, 50, 50000, 0.01)
Y = use(Xtrain, X_means, X_stds, T_means, T_stds, U, V, W)
plt.plot(Xtrain, Ttrain, label='Train')
plt.plot(Xtrain, Y, label='Test')
plt.legend();
plt.show()


