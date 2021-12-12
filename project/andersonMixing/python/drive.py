#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import andMix
import matplotlib.pyplot as plt
import math



global A
n=3
 
def myfun(xin):
	out=np.zeros(shape=(3,1))
	
	x=xin[0];
	y=xin[1];
	z=xin[2];
	
	out[0] = x*y*z-121.;
	out[1] = x*x+y*y-82;
	out[2] = x+y+z-51;
# 	out=A@xin
	return out
 
# x=np.zeros(shape=(3,1))
x=np.random.uniform(-1, 1, size=(n,1))


admix = andMix.andMix()
error_trace=admix.adm_chen (myfun, n, x, 1e-8 , 5000,0.99,10)
print(error_trace)
plt.plot((np.asarray(error_trace)), label='error')
plt.yscale('log')
plt.xlabel('iterations')
plt.ylabel('evaluation error')
# plt.show()
# plt.draw()
fig1 = plt.gcf()
fig1.savefig('Fig1.png',dpi=400)
