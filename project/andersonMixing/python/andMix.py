#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt




class andMix():
	def __init__(self):
		a=1
	
	def adm_chen (self,F, n, x_old, tol, maxIteration,lmd_in,m_in):
		found=0
		kGlobal = 0
		X_end = x_old
		error_s=np.zeros(shape=(1,maxIteration))
		Y = F(X_end)
		err=max(abs(Y))
		print('adm_chen:n=%d Before solving, error=%2.15f \n' % (n,err))
	
		while (err>tol and kGlobal<=maxIteration):
			[X_end, kLocal]=self.mixing(F,n,X_end,tol,kGlobal,maxIteration,lmd_in,m_in)
			kGlobal=kGlobal+kLocal
			Y = F(X_end)
			err=max(abs(Y))
	
		if err<tol:
			found=1
		else:
			print("And_chen failed after %d iterations :(  Try to increase max iteration allowed\n" % (maxIteration));
	
		x_old=X_end;
	
	def mixing(self,F,n,x_old,tol,kGlobal,maxIteration,lmd_in,m_in):
		lmd = lmd_in
		lk = lmd
		nm=m_in
		kLocal = 0
		X=np.zeros((n,maxIteration))
		Y=np.zeros((n,maxIteration))
		X[:,kLocal] = x_old.ravel()
		err=1e99
		U=1
		error_s=np.zeros((maxIteration,1))
		while (err>tol and kLocal+kGlobal-1<=maxIteration):
	
			Y[:,kLocal] = F(X[:,kLocal]).ravel()
			err=max(abs(Y[:,kLocal]))
			if (kLocal+kGlobal>=1):
				error_s[kLocal+kGlobal-1]=err
			
			if kLocal>0: # and (kLocal%10)==0):
				print('adm iteration: %d,n=%d, lk=%e, error: %.14e\n' % (kGlobal+kLocal-1,n,lk,err))
			
			if (err < tol):
				found = 1;
				X_end = X[:,kLocal]
				print('*****And_chen: Solved equation successfully!*****\nThe solution is:\n');
				print(X[:,kLocal])
				return (X_end, kLocal)
	
			if (err > 1e7):
				explode=1
	
	# Calculate the matrix U and the column vector v
			if (kLocal <= nm):
				m = kLocal
			else:
				m = nm
			
			
			U=np.zeros((m,m))
			V=np.zeros((m,1))
			for i in range (0,m):
				V[i,0] = np.dot(Y[:,kLocal] - Y[:,kLocal-i-1],Y[:,kLocal])
				for j in range(m):
					U[i,j] =np.dot(Y[:,kLocal] - Y[:,kLocal-i-1],Y[:,kLocal] - Y[:,kLocal-j-1])
	 
	#Calculate c = U^(-1) * v using Gauss
			
			if (m > 0):
				if np.linalg.cond(U) < 1/1e-14:
					c =np.linalg.solve(U,V) 
				else:
					print("And_chen: Singular Matrix detected And_chen restarted!\n");
					X_end=X[:,kLocal]
					return (X_end, kLocal)
	
	# Calculate the next x^(k)
			for i in range (n):
				cx = 0
				cd = 0
				for j in range(m):
					cx = cx + c[j] * (X[i,kLocal-j-1] - X[i,kLocal])
					cd = cd + c[j] * (Y[i,kLocal-j-1] - Y[i,kLocal])
					
				X[i,kLocal+1] = X[i,kLocal] + cx + (1-lk)*(Y[i,kLocal]+cd)
			
			
			kLocal = kLocal + 1
			if (err<0.03 and kLocal+kGlobal-1>200):  # only modifiy lk if it is close to solution
				lk = lk * lmd
	
			
			if (lk<0.0001):  # reset lk if it is too small
				lk = lmd
	
		X_end=X[:,kLocal]
 



# test function1
# =============================================================================
# def myfun(xin):
# 	out=np.zeros(shape=(3,1))
# 	
# 	x=xin[0];
# 	y=xin[1];
# 	z=xin[2];
# 	
# 	out[0] = x*y*z-121.;
# 	out[1] = x*x+y*y-82;
# 	out[2] = x+y+z-5121;
# 	return out
#  
# x=np.zeros(shape=(3,1))
# x=np.array(np.mat('1;2;3'))
# 
# 
# admix = andMix()
# aa=admix.adm_chen (myfun, 3, x, 1e-12, 50000,0.99,30);
# print(aa)
#  
# 
# =============================================================================













