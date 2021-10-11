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
import qdalda   # from previous lecture



def makeIndicatorVars(T):
	# Make sure T is two-dimensional. Should be nSamples x 1.
	if T.ndim == 1:
		T = T.reshape((-1, 1))    
	return (T == np.unique(T)).astype(int)

T = np.array([1,2,2,1,3]).reshape((-1,1))
T
makeIndicatorVars(T)


def softmax(X, w):
	fs = np.exp(X @ w)  # N x K
	denom = np.sum(fs, axis=1).reshape((-1, 1))
	gs = fs / denom
	return gs

import pandas as pd
data = pd.read_csv('parkinsons.data')
data.shape

X = data
X = X.drop(['status', 'name'], axis=1)
Xnames = X.columns.tolist()
X = X.values

T = data['status'].values
T = T.reshape((-1, 1))
Tname = 'status'

print(X.shape, Xnames, T.shape, Tname)

def standardize(X, mean, stds):
	return (X - mean) / stds

def generate_stratified_partitions(X, T, n_folds, validation=True, shuffle=True):
	'''Generates sets of  Xtrain,Ttrain,Xvalidate,Tvalidate,Xtest,Ttest
	  or
	   sets of Xtrain,Ttrain,Xtest,Ttest if validation is False
	Build dictionary keyed by class label. Each entry contains rowIndices and start and stop
	indices into rowIndices for each of n_folds folds'''

	def rows_in_fold(folds, k):
		all_rows = []
		for c, rows in folds.items():
			class_rows, starts, stops = rows
			all_rows += class_rows[starts[k]:stops[k]].tolist()
		return all_rows

	def rows_in_folds(folds, ks):
		all_rows = []
		for k in ks:
			all_rows += rows_in_fold(folds, k)
		return all_rows

	row_indices = np.arange(X.shape[0])
	if shuffle:
		np.random.shuffle(row_indices)
	folds = {}
	classes = np.unique(T)
	for c in classes:
		class_indices = row_indices[np.where(T[row_indices, :] == c)[0]]
		n_in_class = len(class_indices)
		n_each = int(n_in_class / n_folds)
		starts = np.arange(0, n_each * n_folds, n_each)
		stops = starts + n_each
		stops[-1] = n_in_class
		folds[c] = [class_indices, starts, stops]

	for test_fold in range(n_folds):
		if validation:
			for validate_fold in range(n_folds):
				if test_fold == validate_fold:
					continue
				train_folds = np.setdiff1d(range(n_folds), [test_fold, validate_fold])
				rows = rows_in_fold(folds, test_fold)
				Xtest = X[rows, :]
				Ttest = T[rows, :]
				rows = rows_in_fold(folds, validate_fold)
				Xvalidate = X[rows, :]
				Tvalidate = T[rows, :]
				rows = rows_in_folds(folds, train_folds)
				Xtrain = X[rows, :]
				Ttrain = T[rows, :]
				yield Xtrain, Ttrain, Xvalidate, Tvalidate, Xtest, Ttest
		else:
			# No validation set
			train_folds = np.setdiff1d(range(n_folds), [test_fold])
			rows = rows_in_fold(folds, test_fold)
			Xtest = X[rows, :]
			Ttest = T[rows, :]
			rows = rows_in_folds(folds, train_folds)
			Xtrain = X[rows, :]
			Ttrain = T[rows, :]
			yield Xtrain, Ttrain, Xtest, Ttest

for Xtrain, Ttrain, Xval, Tval, Xtest, Ttest in generate_stratified_partitions(X, T, 4):
	print(f'{len(Ttrain)} {np.mean(Ttrain == 0):.3f} {len(Tval)} {np.mean(Tval == 0):.3f} {len(Ttest)} {np.mean(Ttest == 0):.3f}')
print('\n', np.mean(T == 0))





# =============================================================================
# Linear Logistic model. Logistic network with no hidden layers
# =============================================================================

def runParkLogReg(X, T, n_folds):

	for Xtrain, Ttrain, Xtest, Ttest in generate_stratified_partitions(X, T, n_folds, validation=False):

		means,stds = np.mean(Xtrain, 0), np.std(Xtrain ,0)
		Xtrains = standardize(Xtrain, means, stds)
		Xtests = standardize(Xtest, means, stds)

		Xtrains1 = np.hstack(( np.ones((Xtrains.shape[0], 1)), Xtrains))
		Xtests1 = np.hstack(( np.ones((Xtests.shape[0], 1)), Xtests))

		# New stuff for linear logistic regression

		TtrainI = makeIndicatorVars(Ttrain)
		TtestI = makeIndicatorVars(Ttest)

		w = np.zeros((Xtrains1.shape[1], TtrainI.shape[1]))
		# w = np.random.uniform(size=(Xtrains1.shape[1], TtrainI.shape[1]))
		likelihood = []
		alpha = 0.0001
		for step in range(10000):
			# forward pass
			gs = softmax(Xtrains1, w)
			# backward pass and weight update
			w = w + alpha * Xtrains1.T @ (TtrainI - gs)
			# convert log likelihood to likelihood
			likelihoodPerSample = np.exp( np.sum(TtrainI * np.log(gs)) / Xtrains.shape[0])
			likelihood.append(likelihoodPerSample)

		plt.figure(figsize=(8, 3))
		
		plt.subplot2grid((1, 4), (0, 0))
		plt.plot(likelihood)
		plt.ylabel('Likelihood')
		plt.xlabel('Epoch')

		logregOutput = softmax(Xtrains1, w)
		predictedTrain = np.argmax(logregOutput, axis=1)
		logregOutput = softmax(Xtests1, w)
		predictedTestLR = np.argmax(logregOutput, axis=1)

		print("LogReg: Percent correct: Train {:.3g} Test {:.3g}".format(percentCorrect(predictedTrain, Ttrain),
																		 percentCorrect(predictedTestLR, Ttest)))

		# Previous QDA, LDA code

		qda = qdalda.QDA()
		qda.train(Xtrain, Ttrain)
		qdaPredictedTrain = qda.use(Xtrain)
		qdaPredictedTest = qda.use(Xtest)
		print("   QDA: Percent correct: Train {:.3g} Test {:.3g}".format(percentCorrect(qdaPredictedTrain, Ttrain),
																		 percentCorrect(qdaPredictedTest, Ttest)))

		lda = qdalda.LDA()
		lda.train(Xtrain, Ttrain)
		ldaPredictedTrain = qda.use(Xtrain)
		ldaPredictedTest = qda.use(Xtest)
		print("   LDA: Percent correct: Train {:.3g} Test {:.3g}".format(percentCorrect(ldaPredictedTrain, Ttrain),
																		 percentCorrect(ldaPredictedTest, Ttest)))

		plt.subplot2grid((1, 4), (0, 1), colspan=3)
		plt.plot(Ttest, 'o-', label='Target')
		plt.plot(predictedTestLR, 'o-', label='LR')
		plt.plot(qdaPredictedTest, 'o-', label='QDA')
		plt.plot(ldaPredictedTest, 'o-', label='LDA')
		plt.legend()
		plt.ylabel('Class')
		plt.xlabel('Sample')
		plt.ylim(-0.1, 1.1)
		
		plt.tight_layout()

		break  # only do one data partition

def percentCorrect(p, t):
	return np.sum(p.ravel()==t.ravel()) / float(len(t)) * 100

runParkLogReg(X, T, 5)

exit()

runParkLogReg(X, T, 5)

runParkLogReg(X, T, 5)

import optimizers

# =============================================================================
# Same thing using Adam instead of SGD, However, it defines the error that is used
by SGD
# =============================================================================

def runParkLogReg2(X, T, n_folds):

	for Xtrain, Ttrain, Xtest, Ttest in generate_stratified_partitions(X, T, n_folds, validation=False):

		means,stds = np.mean(Xtrain,0), np.std(Xtrain,0)
		Xtrains = standardize(Xtrain,means,stds)
		Xtests = standardize(Xtest,means,stds)

		Xtrains1 = np.hstack(( np.ones((Xtrains.shape[0],1)), Xtrains))
		Xtests1 = np.hstack(( np.ones((Xtests.shape[0],1)), Xtests))

		TtrainI = makeIndicatorVars(Ttrain)
		TtestI = makeIndicatorVars(Ttest)

		n_classes = TtrainI.shape[1]

		all_weights = np.zeros(Xtrains1.shape[1] * TtrainI.shape[1])
		
		w = all_weights.reshape(( Xtrains1.shape[1], TtrainI.shape[1])) # n_inputs x n_classes

		def softmax(X):
			fs = np.exp(X @ w)  # N x K
			denom = np.sum(fs, axis=1).reshape((-1, 1))
			gs = fs / denom
			return gs

		def neg_log_likelihood():
			# w = warg.reshape((-1,K))
			Y = softmax(Xtrains1)
			return - np.mean(TtrainI * np.log(Y))

		def gradient_neg_log_likelihood():
			Y = softmax(Xtrains1)
			grad = Xtrains1.T @ (Y - TtrainI) / (TtrainI.shape[0] * TtrainI.shape[1])
			return grad.reshape((-1))


		optimizer = optimizers.Optimizers(all_weights)
		to_likelihood = lambda nll: np.exp(-nll)
		
		likelihood_trace = optimizer.adam(neg_log_likelihood, gradient_neg_log_likelihood,
										 n_epochs=10000, learning_rate=0.01, error_convert_f=to_likelihood)


		logregOutput = softmax(Xtrains1)
		predictedTrain = np.argmax(logregOutput,axis=1)
		logregOutput = softmax(Xtests1)
		predictedTest = np.argmax(logregOutput,axis=1)

		print("LogReg: Percent correct: Train {:.3g} Test {:.3g}".format(percentCorrect(predictedTrain,Ttrain),percentCorrect(predictedTest,Ttest)))

		plt.plot(likelihood_trace)
		plt.xlabel('Epoch')
		plt.ylabel('Likelihood')
		
		# Previous QDA code

		qda = qdalda.QDA()
		qda.train(Xtrain, Ttrain)
		qdaPredictedTrain = qda.use(Xtrain)
		qdaPredictedTest = qda.use(Xtest)
		print("   QDA: Percent correct: Train {:.3g} Test {:.3g}".format(percentCorrect(qdaPredictedTrain, Ttrain),
																		 percentCorrect(qdaPredictedTest, Ttest)))

		lda = qdalda.LDA()
		lda.train(Xtrain, Ttrain)
		ldaPredictedTrain = qda.use(Xtrain)
		ldaPredictedTest = qda.use(Xtest)
		print("   LDA: Percent correct: Train {:.3g} Test {:.3g}".format(percentCorrect(ldaPredictedTrain, Ttrain),
																		 percentCorrect(ldaPredictedTest, Ttest)))
		
		break # remove to show all partitioning results

runParkLogReg2(X, T, 8)

runParkLogReg2(X, T, 5)

