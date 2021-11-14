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
import random
import neuralnetworks_A4 as nn

valid_actions = [-1, 0, 1]

def epsilon_greedy(Qnet, state, valid_actions, epsilon):
	'''epsilon is between 0 and 1 and is the probability of returning a random action'''
	
	if np.random.uniform() < epsilon:
		# Random Move
		action = np.random.choice(valid_actions)
		
	else:
		# Greedy Move
		actions_randomly_ordered = random.sample(valid_actions, len(valid_actions))
		Qs = [Qnet.use(np.array([[state, a]])) for a in actions_randomly_ordered]
		ai = np.argmax(Qs)
		action = actions_randomly_ordered[ai]
		
	Q = Qnet.use(np.array([[state, action]]))
	
	return action, Q   # return the chosen action and Q(state, action)


valid_actions = [-1, 0, 1]

def initial_state():
	return np.random.randint(1, 11)

def next_state(s, a):
	newstate = min(max(1, s + a), 10)
	return newstate

def reinf(s, sn):  # sn is next state
	return 0 if sn == 5 else -1  # if we have arrived in state 5, we have reached the goal, so r is 0.

def make_samples(Qnet, initial_state_f, next_state_f, reinforcement_f,
				 valid_actions, n_samples, epsilon):

	X = np.zeros((n_samples, Qnet.n_inputs))
	R = np.zeros((n_samples, 1))
	Qn = np.zeros((n_samples, 1))

	s = initial_state_f()
	a, _ = epsilon_greedy(Qnet, s, valid_actions, epsilon)

	# Collect data from n_samples steps
	for step in range(n_samples):
		
		sn = next_state_f(s, a)        # Update state, sn, from s and a
		rn = reinforcement_f(s, sn)    # Calculate resulting reinforcement
		an, qn = epsilon_greedy(Qnet, sn, valid_actions, epsilon)  # choose next action
		X[step, :] = (s, a)
		R[step, 0] = rn
		Qn[step, 0] = qn
		s, a = sn, an  # Advance one time step


	return (X, R, Qn)


def plot_status():
	n = 100
	xs = np.linspace(1, 10, n).reshape(-1, 1)
	xsas = np.vstack((xs, xs, xs))
	ones = np.ones((n, 1))
	xsas = np.hstack((xsas, np.vstack((ones * -1, ones * 0, ones * 1))))
	ys = Qnet.use(xsas)
	ys = ys.reshape((n, 3), order='F')
	plt.subplot(3, 3, 1)
	plt.plot(np.tile(xs, (1, 3)), ys)
	plt.ylabel('Q for each action')
	plt.xlabel('State')
	plt.legend(('Left','Stay','Right'))

	plt.subplot(3, 3, 4)
	plt.plot(xs, -1 + np.argmax(ys, axis=1))
	plt.ylabel('Action')
	plt.xlabel('State')
	plt.ylim(-1.1, 1.1)

	plt.subplot(3, 3, 7)
	Qnet.use(np.hstack((xs, ones * -1)))
	plt.gca().set_prop_cycle(None)
	plt.plot(xs, Qnet.Ys[1], '--')
	Qnet.use(np.hstack((xs, ones * 0)))
	plt.gca().set_prop_cycle(None)
	plt.plot(xs, Qnet.Ys[1])  # Ys[0] is inputs, Ys[1] is hidden layer, Ys[2] is output layer
	Qnet.use(np.hstack((xs, ones * 1)))
	plt.plot(xs, Qnet.Ys[1], linewidth=2)                   
	plt.ylabel('Hidden Units for Actions -1 (dotted) 0 (solid) 1 (thick)')
	plt.xlabel('State')

	plt.subplot(3, 3, 2)
	x = x_trace[j - 200:j, 0]
	plt.plot(x, range(200), '.-')
	plt.xlabel('x (last 200)')
	plt.ylabel('Steps')
	y = x.copy()
	y[y != 5] = np.nan
	plt.plot(y, range(200), 'ro')
	plt.xlim(0, 10)

	plt.subplot(3, 3, 5)
	plt.plot(epsilon_trace[:trial])
	plt.ylabel('$\epsilon$')
	plt.xlabel('Trials')
	plt.ylim(0, 1)

	plt.subplot(3, 3 ,8)
	plt.plot(error_trace)
	plt.ylabel('TD Error')
	plt.xlabel('Epochs')
					
	plt.subplot(3, 3, 3)
	plt.plot(r_trace[:j, 0])
	plt.ylim(-1.1, 0.1)
	plt.ylabel('R')
	plt.xlabel('Steps')
					
	plt.subplot(3, 3, 6)
	plt.plot(np.convolve(r_trace[:j, 0], np.array([0.01] * 100), mode='valid'))
	plt.ylabel('R smoothed')
	plt.ylim(-1.1, 0)
	plt.xlabel('Steps')

	# plt.subplot(3, 3, 9)
	# Qnet.draw(('x', 'a'), ('Q'))
	
	plt.tight_layout()


n_trials = 5000
n_steps_per_trial = 20
n_epochs = 20
learning_rate = 0.1

n_hidden = [10]
gamma = 0.8
final_epsilon = 0.001  # value of epsilon at end of simulation. Decay rate is calculated

epsilon_decay =  np.exp(np.log(final_epsilon) / n_trials) # to produce this final value
print('epsilonDecay is', epsilon_decay)

from IPython.display import display, clear_output

epsilon = 1.0

Qnet = nn.NeuralNetwork(2, n_hidden, 1)

# We need to set standardization parameters now so Qnet can be called to get first set of samples,
# before it has been trained the first time.

def setup_standardization(Qnet, Xmeans, Xstds, Tmeans, Tstds):
	Qnet.X_means = np.array(Xmeans)
	Qnet.X_stds = np.array(Xstds)
	Qnet.T_means = np.array(Tmeans)
	Qnet.T_stds = np.array(Tstds)

# Inputs are position (1 to 10) and action (-1, 0, or 1)
setup_standardization(Qnet, [5, 0], [2.5, 0.5], [0], [1])

fig = plt.figure(figsize=(10, 10))

x_trace = np.zeros((n_trials * n_steps_per_trial, 2))
r_trace = np.zeros((n_trials * n_steps_per_trial, 1))
error_trace = []
epsilon_trace = np.zeros((n_trials, 1))

for trial in range(n_trials):
	
	X, R, Qn = make_samples(Qnet, initial_state, next_state, reinf, valid_actions, n_steps_per_trial, epsilon)
 
	T = R + gamma * Qn
	Qnet.train(X, T, n_epochs, method='sgd', learning_rate=learning_rate, batch_size=-1, verbose=False)
	
	epsilon_trace[trial] = epsilon
	i = trial * n_steps_per_trial
	j = i + n_steps_per_trial
	x_trace[i:j, :] = X
	r_trace[i:j, :] = R
	error_trace += Qnet.error_trace
	
	epsilon *= epsilon_decay

	# Rest of this loop is for plots.
	if True and (trial + 1) % int(n_trials * 0.01 + 0.5) == 0:
		
		fig.clf()
		plot_status()
		clear_output(wait=True)
		display(fig)
	
clear_output(wait=True)


def run(n_trials, n_steps_per_trial, n_epochs, learning_rate, n_hidden, gamma, final_epsilon):
	epsilon = 1
	epsilon_decay =  np.exp(np.log(final_epsilon)/n_trials) # to produce this final value

	Qnet = nn.NeuralNetwork(2, n_hidden, 1)
	setup_standardization(Qnet, [5, 0], [2.5, 0.5], [0], [1])

	r_sum = 0
	r_last_2_trials = 0
	for trial in range(n_trials):
		X, R, Qn = make_samples(Qnet, initial_state, next_state, reinf, valid_actions, n_steps_per_trial, epsilon)
		r_sum += np.sum(R)
		if trial > n_trials - 3:
			r_last_2_trials += np.sum(R)
		epsilon *= epsilon_decay
		Qnet.train(X, R + gamma * Qn, n_epochs, method='sgd', learning_rate=learning_rate, batch_size=-1, verbose=False)
		
	return r_sum / (n_trials * n_steps_per_trial), r_last_2_trials / (2 * n_steps_per_trial)

import time
start = time.time()
result = []
for nT in (10, 50, 100):  # number of trials
	print('nT is', nT)
	for nS in (10, 20, 50, 100):  # number of steps per trial
		for nE in (5, 10, 50, 100):  # number of epochs
			for nH in [[2], [5], [10], [10, 10]]:  # hidden layer structure
				for g in (0.5, 0.9):  # gamma
					for lr in (0.01, 0.1):  # learning rate
						for fep in (0.01,):  # final epsilon
							
							r, r_last_2 = run(nT, nS, nE, lr, nH, g, fep)
							result.append([nT, nS, nE, nH, g, lr, fep, r, r_last_2])
							# print(result[-1])
							
print(f'Took {(time.time() - start)/60:.1f} minutes.')


import pandas
result = pandas.DataFrame(result, columns=('Trials', 'StepsPerTrial', 'Epochs',
										   'Hiddens', 'gamma', 'lr', 'fep', 'R', 'R Last 2'))
result.sort_values(by='R Last 2', ascending=False)
