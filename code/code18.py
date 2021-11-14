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
import matplotlib.pyplot as plt
# %matplotlib notebook  # We don't need the interaction with our plots in this notebook
import pickle
import gzip

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import neuralnetworks_A4 as nn   # from A4

valid_actions = (-1, 0, 1)   # Possible actions

def reinforcement(s, sn):
	goal = 5
	return 0 if abs(sn[0 ]- goal) < 1 else -1
	# return -abs(sn[0] - goal)

def initial_state():
	return np.array([10 * np.random.uniform(), 0.0])

def next_state(s, a):
	'''[0] is position, s[1] is velocity. a is -1, 0 or 1'''    
	s = s.copy()
	deltaT = 0.1                           # Euler integration time step
	s[0] += deltaT * s[1]                  # Update position
	force = a
	mass = 0.5
	s[1] += deltaT * (force / mass - 0.2 * s[1])  # Update velocity. Includes friction

	# Bound next position. If at limits, set velocity to 0.
	if s[0] < 0:        
		s = [0., 0.]    # these constants as ints were causing the errors we discussed in class. I DON'T KNOW WHY!!
	elif s[0] > 10:
		s = [10., 0.]
		
	return np.array(s)

s = initial_state()
pos = []
for i in range(1000):
	a = np.random.choice(valid_actions)
	s = next_state(s, a)
	r = reinforcement(s, s)
	pos.append(s.tolist() + [a, r])

plt.plot(np.array(pos))
plt.legend(('pos', 'vel', 'a', 'r'))

s = initial_state()
sn = next_state(s, -1)
reinforcement(s, sn)

def epsilon_greedy(Qnet, state, valid_actions, epsilon):
	
	if np.random.uniform() < epsilon:
		# Random Move
		action = np.random.choice(valid_actions)
		
	else:
		# Greedy Move
		Qs = [Qnet.use(np.hstack((state, a)).reshape((1, -1))) for a in valid_actions]
		ai = np.argmax(Qs)
		action = valid_actions[ai]
		
	Q = Qnet.use(np.hstack((state, action)).reshape((1, -1)))
	
	return action, Q

def make_samples(Qnet, initial_state_f, next_state_f, reinforcement_f, valid_actions, n_samples, epsilon):

	X = np.zeros((n_samples, Qnet.n_inputs))
	R = np.zeros((n_samples, 1))
	Qn = np.zeros((n_samples, 1))

	s = initial_state_f()
	s = next_state_f(s, 0)        # Update state, sn from s and a
	a, _ = epsilon_greedy(Qnet, s, valid_actions, epsilon)

	# Collect data from n_samples steps
	for step in range(n_samples):
		
		sn = next_state_f(s, a)        # Update state, sn from s and a
		rn = reinforcement_f(s, sn)   # Calculate resulting reinforcement
		an, qn = epsilon_greedy(Qnet, sn, valid_actions, epsilon) # Forward pass for time t+1
		X[step, :] = np.hstack((s, a))
		R[step, 0] = rn
		Qn[step, 0] = qn
		# Advance one time step
		s, a = sn, an

	return (X, R, Qn)

from matplotlib import cm

def plot_status(Qnet, trial, epsilon_trace, r_trace):
	
	plt.subplot(3, 3, 1)
	plt.plot(epsilon_trace[:trial + 1])
	plt.ylabel('Random Action Probability ($\epsilon$)')
	plt.ylim(0, 1)
			   
	plt.subplot(3, 3, 2)
	plt.plot(r_trace[:trial + 1], alpha=0.5)
	binSize = 20
	if trial+1 > binSize:
		# Calculate mean of every bin of binSize reinforcement values
		smoothed = np.mean(r_trace[:int(trial / binSize) * binSize].reshape((int(trial / binSize), binSize)), axis=1)
		plt.plot(np.arange(1, 1 + int(trial / binSize)) * binSize, smoothed)
	plt.ylabel('Mean reinforcement')
		
	qs = Qnet.use(np.array([[s, 0, a] for a in valid_actions for s in range(11)]))
			   
	plt.subplot(3, 3, 3)
	acts = ['L', '0', 'R']
	actsiByState = np.argmax(qs.reshape((len(valid_actions), -1)), axis=0)
	for i in range(11):
		plt.text(i, 0, acts[actsiByState[i]])
		plt.xlim(-1, 11)
		plt.ylim(-1, 1)
	plt.text(2, 0.2,'Policy for Zero Velocity')
	plt.axis('off')
			   
			   
	plt.subplot(3, 3, 4)
	n = 20
	positions = np.linspace(0, 10, n)
	velocities =  np.linspace(-5, 5, n)
	xs, ys = np.meshgrid(positions, velocities)
	xsflat = xs.flat
	ysflat = ys.flat
	qs = Qnet.use(np.array([[xsflat[i], ysflat[i], a] for a in valid_actions for i in range(len(xsflat))]))
	qs = qs.reshape((len(valid_actions), -1)).T
	qsmax = np.max(qs, axis=1).reshape(xs.shape)
	cs = plt.contourf(xs, ys, qsmax)
	plt.colorbar(cs)
	plt.xlabel('$x$')
	plt.ylabel('$\dot{x}$')
	plt.title('Max Q')
			   
	plt.subplot(3, 3, 5)
	acts = np.array(valid_actions)[np.argmax(qs, axis=1)].reshape(xs.shape)
	cs = plt.contourf(xs, ys, acts, [-2, -0.5, 0.5, 2])
	plt.colorbar(cs)
	plt.xlabel('$x$')
	plt.ylabel('$\dot{x}$')
	plt.title('Actions')
	
	plt.subplot(3, 3, 6)
	plt.plot(X[:, 0], X[: ,1])
	plt.plot(X[-1, 0], X[-1, 1], 'ro')
	plt.xlabel('$x$')
	plt.ylabel('$\dot{x}$')
	plt.fill_between([4, 6], [-5, -5], [5, 5], color='red', alpha=0.3)  # CHECK OUT THIS FUNCTION!
	plt.xlim(-1, 11)
	plt.ylim(-5, 5)
	plt.title('Last Trial')

	ax = plt.subplot(3, 3, 7, projection='3d')
	ax.plot_surface(xs, ys, qsmax, linewidth=0)  #cmap=cm.jet, linewidth=0)
	ax.set_xlabel('$x$')
	ax.set_ylabel('$\dot{x}$')
	plt.title('Max Q')
	
	ax = plt.subplot(3, 3, 8, projection='3d')
	ax.plot_surface(xs, ys, acts, cmap=cm.jet, linewidth=0)
	ax.set_xlabel('$x$')
	ax.set_ylabel('$\dot{x}$')
	plt.title('Action')
	
	plt.tight_layout()

def test_it(Qnet, n_trials, n_steps_per_trial):
	xs = np.linspace(0, 10, n_trials)
	plt.subplot(3, 3, 9) 
	
	# For a number (n_trials) of starting positions, run marble sim for n_steps_per_trial
	for x in xs:
		
		s = [x, 0]  # 0 velocity
		x_trace = np.zeros((n_steps_per_trial, 2))
		for step in range(n_steps_per_trial):
			a, _ = epsilon_greedy(Qnet, s, valid_actions, 0.0) # epsilon = 0
			s = next_state(s, a)
			x_trace[step, :] = s
			
		plt.plot(x_trace[:, 0], x_trace[:, 1])
		plt.plot(x_trace[-1, 0], x_trace[-1, 1], 'ro')
		plt.fill_between([4, 6], [-5, -5], [5, 5], color='pink', alpha=0.3)
		plt.xlim(-1, 11)
		plt.ylim(-5, 5)
		plt.ylabel('$\dot{x}$')
		plt.xlabel('$x$')
		plt.title('State Trajectories for $\epsilon=0$')
		

def setup_standardization(Qnet, Xmeans, Xstds, Tmeans, Tstds):
	Qnet.X_means = np.array(Xmeans)
	Qnet.X_stds = np.array(Xstds)
	Qnet.T_means = np.array(Tmeans)
	Qnet.T_stds = np.array(Tstds)

np.std([-1, 1]), np.std(np.arange(10))

from IPython.display import display, clear_output
fig = plt.figure(figsize=(10, 10))

gamma = 0.8        # discount factor
n_trials = 2000        # number of repetitions of makeSamples-updateQ loop
n_steps_per_trial = 50 # number of steps between new random initial states
n_epochs = 30
learning_rate = 0.05
final_epsilon = 0.01 # value of epsilon at end of simulation. Decay rate is calculated
epsilon_decay =  np.exp(np.log(final_epsilon) / (n_trials)) # to produce this final value
print('epsilon_decay is', epsilon_decay)
epsilon_trace = np.zeros(n_trials)
r_trace = np.zeros(n_trials)

# Create the Qnet
nh = [5, 5]  # hidden layers structure
Qnet = nn.NeuralNetwork(3, nh, 1)
# Inputs are position (1 to 10) velocity (-3 to 3) and action (-1, 0, or 1)
setup_standardization(Qnet, [5, 0, 0], [2, 2, 0.8], [-2], [1])

epsilon = 1         # initial epsilon value

# Train for n_trials
for trial in range(n_trials):
	
	# Collect n_steps_per_trial samples of X, R, Qn, and Q, and update epsilon
	X, R, Qn = make_samples(Qnet, initial_state, next_state, reinforcement, valid_actions, 
							n_steps_per_trial, epsilon)
	
	Qnet.train(X, R + gamma * Qn, n_epochs,  method='sgd', learning_rate=learning_rate, batch_size=-1, verbose=False)

	# Rest is for plotting
	epsilon_trace[trial] = epsilon
	epsilon *= epsilon_decay
	r_trace[trial] = np.mean(R)
	if True and (trial + 1 == n_trials or trial % (n_trials / 50) == 0):
		fig.clf()
		plot_status(Qnet, trial, epsilon_trace, r_trace)
		test_it(Qnet, 10, 500)
		clear_output(wait=True)
		display(fig)
	
clear_output(wait=True)

import matplotlib.patches as pltpa 
from IPython.display import display, clear_output
fig = plt.figure(figsize=(10, 2))

for trial in range(10):
	x = np.random.uniform(0, 10)
	s = [x, 0]  # 0 velocity
	for step in range(n_steps_per_trial):
		plt.clf()
		x = s[0]
		a, _ = epsilon_greedy(Qnet, s, valid_actions, 0.0) # epsilon = 0
		plt.plot([0, 10], [0, 0], lw=3, color='gray')
		ax = fig.gca()
		ax.add_patch(pltpa.Rectangle((4, 0), 2, 1, color='pink'))
		ax.add_patch(pltpa.Rectangle((x-0.5, 0), 1, 1, color='blue'))
		if a != 0:
			plt.arrow(x, 0.5 , a * 0.4, 0., head_width=0.1, head_length=0.1, color='yellow')
		plt.xlim(-1, 11)
		plt.ylim(0, 2)
		plt.axis('off')
		
		s = next_state(s, a)

		clear_output(wait=True)
		display(fig)
	
clear_output(wait=True)


 