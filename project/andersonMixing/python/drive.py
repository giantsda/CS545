#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import andMix


global A
n=6
A=np.random.uniform(-1, 1, size=(n, n))
def myfun(xin):
	# out=np.zeros(shape=(1,1))
	
	x=xin[0];
	y=xin[1];
 
	
	# out[0] = x*y*z-121.;
	# out[1] = x*x+y*y-82;
	# out[2] = x+y+z-5121;
	out=A@xin
	return out
 
# x=np.zeros(shape=(3,1))
x=np.random.uniform(-1, 1, size=(n,1))


admix = andMix.andMix()
x_end=admix.adm_chen (myfun, n, x, 1e-12, 5000,0.99,10)
print(x_end)
 
 
