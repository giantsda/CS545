#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import andMix


def myfun(xin):
	# out=np.zeros(shape=(1,1))
	
	x=xin[0];
	y=xin[1];
 
	
	# out[0] = x*y*z-121.;
	# out[1] = x*x+y*y-82;
	# out[2] = x+y+z-5121;
	out=(x-1)*(x-1)+(y-2)*(y-3)-5
	return np.array([out,0])
 
# x=np.zeros(shape=(3,1))
x=np.array(np.mat('1;1.21113456'))


admix = andMix.andMix()
admix.adm_chen (myfun, 2 , x, 1e-12, 50000,0.99,30);
 
 
