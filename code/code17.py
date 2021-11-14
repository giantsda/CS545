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
import copy

def printBoard(board):
	print('''
{}|{}|{}
-----
{}|{}|{}
------
{}|{}|{}'''.format(*tuple(board)))
printBoard(np.array(['X',' ','O', ' ','X','O', 'X',' ',' ']))

def winner(board):
	combos = np.array((0,1,2, 3,4,5, 6,7,8, 0,3,6, 1,4,7, 2,5,8, 0,4,8, 2,4,6))
	if np.any(np.logical_or(np.all('X' == board[combos].reshape((-1, 3)), axis=1),
							np.all('O' == board[combos].reshape((-1, 3)), axis=1))):
		return True
	else:
		return False

board = np.array(['X',' ','O', ' ','X','O', 'X',' ',' '])
printBoard(board), print(winner(board))
board = np.array(['X',' ','X', ' ','X','O', 'X',' ',' '])
printBoard(board), print(winner(board))

np.where(board == ' ')

np.where(board == ' ')[0]

board = np.array(['X',' ','O', ' ','X','O', 'X',' ',' '])
validMoves = np.where(board == ' ')[0]
move = np.random.choice(validMoves)
boardNew = copy.copy(board)
boardNew[move] = 'X'
print('From this board')
printBoard(board)
print('\n  Move',move)
print('\nresults in board')
printBoard(boardNew)

Q = {}  # empty table
Q[(tuple(board), 1)] = 0
 

Q[(tuple(board), 1)]

board[1] = 'X'
Q[(tuple(board), 1)]

Q.get((tuple(board), 1), 42)

Q[(tuple(board), move)] = 1
Q[(tuple(board), move)] = 0

rho = 0.1 # learning rate
Q[(tuple(board), move)] += rho * (-1 - Q[(tuple(board), move)])

step = 0
if step > 0:
	Q[(tuple(boardOld), moveOld)] += rho * (Q[(tuple(board), move)] - Q[(tuple(boardOld), moveOld)])

validMoves = np.where(board == ' ')[0]
print('Valid moves are', validMoves)
Qs = np.array([Q.get((tuple(board), m), 0) for m in validMoves]) 
print('Q values for validMoves are', Qs)
bestMove = validMoves[np.argmax(Qs)]
print('Best move is', bestMove)

s=np.array([4, 2, 5])
np.random.shuffle(s)

def epsilonGreedy(epsilon, Q, board):
	validMoves = np.where(board == ' ')[0]
	if np.random.uniform() < epsilon:
		# Random Move
		return np.random.choice(validMoves)
	else:
		# Greedy Move
		np.random.shuffle(validMoves)
		Qs = np.array([Q.get((tuple(board) ,m), 0) for m in validMoves]) 
		return validMoves[ np.argmax(Qs) ]
	
epsilonGreedy(0.8, Q, board)

outcomes = np.random.choice([-1, 0, 1], replace=True, size=(1000))
outcomes[:10]

def plotOutcomes(outcomes, epsilons, maxGames, nGames):
	if nGames == 0:
		return
	nBins = 100
	nPer = maxGames // nBins
	outcomeRows = outcomes.reshape((-1, nPer))
	outcomeRows = outcomeRows[:nGames // nPer + 1, :]
	avgs = np.mean(outcomeRows, axis=1)
	
	plt.subplot(3, 1, 1)
	xs = np.linspace(nPer, nGames, len(avgs))
	plt.plot(xs, avgs)
	plt.xlabel('Games')
	plt.ylabel('Mean of Outcomes\n(0=draw, 1=X win, -1=O win)')
	plt.title('Bins of {:d} Games'.format(nPer))
	
	plt.subplot(3, 1, 2)
	plt.plot(xs,np.sum(outcomeRows==-1, axis=1), 'r-', label='Losses')
	plt.plot(xs,np.sum(outcomeRows==0,axis=1),'b-', label='Draws')
	plt.plot(xs,np.sum(outcomeRows==1,axis=1), 'g-', label='Wins')
	plt.legend(loc="center")
	plt.ylabel('Number of Games\nin Bins of {:d}'.format(nPer))
	
	plt.subplot(3, 1, 3)
	plt.plot(epsilons[:nGames])
	plt.ylabel('$\epsilon$')

plt.figure(figsize=(8, 8))
plotOutcomes(outcomes, np.zeros(1000), 1000, 1000)

from IPython.display import display, clear_output


# =============================================================================
# test
# =============================================================================

maxGames = 10000
rho = 0.5
epsilonDecayRate = 0.999
epsilon = 1.0
graphics = True
showMoves = not graphics

outcomes = np.zeros(maxGames)
epsilons = np.zeros(maxGames)
Q = {}

if graphics:
	fig = plt.figure(figsize=(10, 10))

for nGames in range(maxGames):
	
	epsilon *= epsilonDecayRate
	epsilons[nGames] = epsilon
	step = 0
	board = np.array([' '] * 9)  # empty board
	done = False
	
	while not done:        
		step += 1
		
		# X's turn
		move = epsilonGreedy(epsilon, Q, board)
		boardNew = copy.copy(board)
		boardNew[move] = 'X'
		if (tuple(board), move) not in Q:
			Q[(tuple(board), move)] = 0  # initial Q value for new board,move
		if showMoves:
			printBoard(boardNew)
			
		if winner(boardNew):
			# X won!
			if showMoves:
				print('        X Won!')
			Q[(tuple(board), move)] = 1
			done = True
			outcomes[nGames] = 1
			
		elif not np.any(boardNew == ' '):
			# Game over. No winner.
			if showMoves:
				print('        draw.')
			Q[(tuple(board), move)] = 0
			done = True
			outcomes[nGames] = 0
			
		else:
			# O's turn.  O is a random player!
			moveO = np.random.choice(np.where(boardNew==' ')[0])
			boardNew[moveO] = 'O'
			if showMoves:
				printBoard(boardNew)
			if winner(boardNew):
				# O won!
				if showMoves:
					print('        O Won!')
				Q[(tuple(board), move)] += rho * (-1 - Q[(tuple(board), move)])
				done = True
				outcomes[nGames] = -1
		
		if step > 1:
			Q[(tuple(boardOld), moveOld)] += rho * (Q[(tuple(board), move)] - Q[(tuple(boardOld), moveOld)])
			
		boardOld, moveOld = board, move # remember board and move to Q(board,move) can be updated after next steps
		board = boardNew
		
		if graphics and (nGames % (maxGames/10) == 0 or nGames == maxGames-1):
			fig.clf() 
			plotOutcomes(outcomes, epsilons ,maxGames, nGames-1)
			clear_output(wait=True)
			display(fig);

if graphics:
	clear_output(wait=True)
print('Outcomes: {:d} X wins {:d} O wins {:d} draws'.format(np.sum(outcomes==1), np.sum(outcomes==-1), np.sum(outcomes==0)))


Q[(tuple([' ']*9),0)]

Q[(tuple([' ']*9),1)]
Q.get((tuple([' ']*9),0), 0)

[Q.get((tuple([' ']*9),m), 0) for m in range(9)]

board = np.array([' ']*9)
Qs = [Q.get((tuple(board),m), 0) for m in range(9)]
printBoard(board)
print('''{:.2f} | {:.2f} | {:.2f}
------------------
{:.2f} | {:.2f} | {:.2f}
------------------
{:.2f} | {:.2f} | {:.2f}'''.format(*Qs))

def printBoardQs(board,Q):
	printBoard(board)
	Qs = [Q.get((tuple(board),m), 0) for m in range(9)]
	print()
	print('''{:.2f} | {:.2f} | {:.2f}
------------------
{:.2f} | {:.2f} | {:.2f}
------------------
{:.2f} | {:.2f} | {:.2f}'''.format(*Qs))


board[0] = 'X'
board[1] = 'O'
printBoardQs(board,Q)

board[4] = 'X'
board[3] = 'O'
printBoardQs(board,Q)

board = np.array([' ']*9)
printBoardQs(board,Q)

board[0] = 'X'
board[4] = 'O'
printBoardQs(board,Q)

board[2] = 'X'
board[1] = 'O'
printBoardQs(board,Q)


# =============================================================================
# Neural Network as Q function for Tic-Tac-Toe
# =============================================================================


import neuralnetworks_A4 as nn


def initial_state():
	return np.array([0] * 9)

def next_state(s, a, marker):  # s is a board, and a is an index into the cells of the board, marker is 1 or -1
	s = s.copy()
	s[a] = 1 if marker == 'X' else -1
	return s

def reinforcement(s):
	if won('X', s):
		return 1
	if won('O', s):
		return -1
	return 0

def won(player, s):
	marker = 1 if player == 'X' else -1
	combos = np.array((0,1,2, 3,4,5, 6,7,8, 0,3,6, 1,4,7, 2,5,8, 0,4,8, 2,4,6))
	return np.any(np.all(marker == s[combos].reshape((-1, 3)), axis=1))

def draw(s):
	return sum(s == 0) == 0

def valid_actions(state):
	return np.where(state == 0)[0]

def stack_sa(s, a):
	return np.hstack((s, a)).reshape(1, -1)

def other_player(player):
	return 'X' if player == 'O' else 'O'

def epsilon_greedy(Qnet, state, epsilon):
	
	actions = valid_actions(state)
	
	if np.random.uniform() < epsilon:
		# Random Move
		action = np.random.choice(actions)
		
	else:
		# Greedy Move
		np.random.shuffle(actions)
		Qs = np.array([Qnet.use(stack_sa(state, a)) for a in actions])
		action = actions[np.argmax(Qs)]
		
	return action

def make_samples(Qnets, initial_state_f, next_state_f, reinforcement_f, epsilon):
	'''Run one game'''
	X = []
	R = []
	Qn = []

	s = initial_state_f()
	player = 'X'

	while True:
		
		a = epsilon_greedy(Qnets[player], s, epsilon)
		sn = next_state_f(s, a, player)
		r = reinforcement_f(s)

		X.append(stack_sa(s, a))
		R.append(r)

		if r != 0 or draw(sn):
			break

		s = sn
		player = other_player(player)  # switch

	X = np.vstack(X)
	R = np.array(R).reshape(-1, 1)

	# Assign all Qn's, based on following state, but go every other state to do all X values,
	# and to do all O values.
	Qn = np.zeros_like(R)
	if len(Qn) % 2 == 1:
		# Odd number of samples, so 0 won
		# for X samples
		Qn[:-4:2, :] = Qnets['X'].use(X[2:-2:2])  # leave last sample Qn=0
		R[-2, 0] = R[-1, 0]  # copy final r (win for O) to last X state, too
		# for O samples
		Qn[1:-4:2, :] = Qnets['O'].use(X[3:-2:2])  # leave last sample Qn=0
	else:
		# Odd number of samples, so X won or draw
		# for X samples
		Qn[:-4:2, :] = Qnets['X'].use(X[2:-2:2])  # leave last sample Qn=0
		R[-2, 0] = - R[-1, 0]  # copy negated final r (win for X) to last O state, too
		# for O samples
		Qn[1:-4:2, :] = Qnets['O'].use(X[3:-2:2])
		
	return {'X': X, 'R': R, 'Qn': Qn}



def plot_status(outcomes, epsilons, n_trials, trial):
	if trial == 0:
		return
	outcomes = np.array(outcomes)
	n_per = 10
	n_bins = (trial + 1) // n_per
	if n_bins == 0:
		return
	outcome_rows = outcomes[:n_per * n_bins].reshape((-1, n_per))
	outcome_rows = outcome_rows[:trial // n_per + 1, :]
	avgs = np.mean(outcome_rows, axis=1)
	
	plt.subplot(3, 1, 1)
	xs = np.linspace(n_per, n_per * n_bins, len(avgs))
	plt.plot(xs, avgs)
	plt.ylim(-1.1, 1.1)
	plt.xlabel('Games')
	plt.ylabel('Mean of Outcomes') # \n(0=draw, 1=X win, -1=O win)')
	plt.title(f'Bins of {n_per:d} Games')
	
	plt.subplot(3, 1, 2)
	plt.plot(xs, np.sum(outcome_rows == -1, axis=1), 'r-', label='Losses')
	plt.plot(xs, np.sum(outcome_rows == 0, axis=1), 'b-', label='Draws')
	plt.plot(xs, np.sum(outcome_rows == 1, axis=1), 'g-', label='Wins')
	plt.legend(loc='center')
	plt.ylabel(f'Number of Games\nin Bins of {n_per:d}')
	
	plt.subplot(3, 1, 3)
	plt.plot(epsilons[:trial])
	plt.ylabel('$\epsilon$')


def setup_standardization(Qnet, Xmeans, Xstds, Tmeans, Tstds):
	Qnet.X_means = np.array(Xmeans)
	Qnet.X_stds = np.array(Xstds)
	Qnet.T_means = np.array(Tmeans)
	Qnet.T_stds = np.array(Tstds)


from IPython.display import display, clear_output
fig = plt.figure(figsize=(10, 10))

gamma = 0.8       # discount factor
n_trials = 500         # number of repetitions of makeSamples-updateQ loop
n_epochs = 5
learning_rate = 0.01
final_epsilon = 0.01 # value of epsilon at end of simulation. Decay rate is calculated
epsilon_decay =  np.exp(np.log(final_epsilon) / (n_trials)) # to produce this final value
# epsilon_decay = 1  # to force both players to take random actions
print('epsilon_decay is', epsilon_decay)

#################################################################################
# Qnet for Player 'X'
nhX = [5]  # hidden layers structure
QnetX = nn.NeuralNetwork(9 + 1, nhX, 1)

# Qnet for Player 'O'
nhO = []  # hidden layers structure
QnetO = nn.NeuralNetwork(9 + 1, nhO, 1)
#################################################################################

# Inputs are 9 TTT cells plus 1 action
setup_standardization(QnetX, [0] * 10, [1] * 10, [0], [1])
setup_standardization(QnetO, [0] * 10, [1] * 10, [0], [1])

Qnets = {'X': QnetX, 'O': QnetO}

fig = plt.figure(1, figsize=(10, 10))

epsilon = 1         # initial epsilon value
outcomes = []
epsilon_trace = []

# Train for n_trials
for trial in range(n_trials):
	
	samples = make_samples(Qnets, initial_state, next_state, reinforcement, epsilon)
	
	for player in ['X', 'O']:
		first_sample = 0 if player == 'X' else 1
		rows = slice(0, None, 2) if player == 'X' else slice(1, None, 2)
		X = samples['X'][rows, :]
		R = samples['R'][rows, :]
		Qn = samples['Qn'][rows, :]
		T = R + gamma * Qn
		Qnets[player].train(X, T, n_epochs, method='sgd', learning_rate=learning_rate, batch_size=-1, verbose=False)

	# Rest is for plotting
	epsilon_trace.append(epsilon)
	epsilon *= epsilon_decay
	n_moves = len(samples['R'])
	final_r = samples['R'][-1]
	# if odd n_moves, then O won so negate final_r for X perspective
	outcome = final_r if n_moves % 2 == 0 else -final_r
	outcomes.append(outcome)
	if True and (trial + 1 == n_trials or trial % (n_trials / 20) == 0):
		fig.clf()
		plot_status(outcomes, epsilon_trace, n_trials, trial)
		clear_output(wait=True)
		display(fig)

clear_output(wait=True);



 