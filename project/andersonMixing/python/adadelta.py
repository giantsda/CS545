#!/usr/bin/python3

# example of plotting the adadelta search on a contour plot of the test function
from math import sqrt
from numpy import asarray
from numpy import arange
import  numpy as np
from numpy.random import rand
from numpy.random import seed
from numpy import meshgrid
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import time


# objective function
def function(x, y):
	return x**2.0 + y**2.0
 
# derivative of objective function
def derivative(x, y):
	return asarray([x * 2.0, y * 2.0])
 
# gradient descent algorithm with adadelta
def adadelta(function, gradient, n_iter, rho):
	# Based on https://d2l.ai/chapter_optimization/adadelta.html
	ep=1e-3
	solutions = list()
	solution = np.random.uniform(-1, 1, size=(2))
	St = np.zeros(shape=2)
	delta = np.zeros(shape=2)
	# Main loop
	for it in range(n_iter):
		dW = gradient(solution[0], solution[1])
		St=St* rho + (dW**2.0 * (1.0-rho))
		rescaledGradient = np.sqrt(delta+ep)/np.sqrt(St+ep)*dW
		solution = solution - rescaledGradient
		delta=delta * rho + rescaledGradient**2.0 * (1.0-rho)
		
		solutions.append(solution)
		print('>%d f(%s) = %.5f' % (it, solution, function(solution[0], solution[1])))
	return solutions
 

for i in range (100):
	# seed the pseudo random number generator
	np.random.seed(int(time.time()))
	# np.random.seed(1)
	# define range for input
	bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
	# define the total iterations
	n_iter = 120
	# rho for adadelta
	rho = 0.9
	# perform the gradient descent search with adadelta
	solutions = adadelta(function, derivative, n_iter, rho)
	# sample input range uniformly at 0.1 increments
	xaxis = arange(bounds[0,0], bounds[0,1], 0.1)
	yaxis = arange(bounds[1,0], bounds[1,1], 0.1)
	# create a mesh from the axis
	x, y = meshgrid(xaxis, yaxis)
	# compute targets
	results = function(x, y)
	# create a filled contour plot with 50 levels and jet color scheme
	pyplot.contourf(x, y, results, levels=50, cmap='jet')
	# plot the sample as black circles
	solutions = np.asarray(solutions)
	pyplot.plot(solutions[:, 0], solutions[:, 1], '.-', color='w')
	# show the plot
	pyplot.show()