//============================================================================
// Name        : sandbox.cpp
// Author      : Chen
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================


#include <vector>  // std::vector
#include "nr.h"  //gaussj
#include <stdio.h> // printf()
#include <cmath>        // std::abs

using namespace std;

int
adm_chen (void
(*f) (vector<double>& in, vector<double>& out),
	  vector<double>& x_old, double tol, int maxIteration)
{
  int k = 0; // kth iteration
  double lmd = 0.9; // relaxzation factor
  double lk = lmd;
  int m, n = x_old.size (); // n is the size of the vector problem
  int nm = min (30, n);
  int k_restart = 0; // k_restart is used when U is ill.
  double err = 9.9e99; // err
  vector<vector<double> > X (maxIteration), Y (maxIteration); /* X is used to store the guessed solution and
   Y is the resulted rhs*/
  for (int i = 0; i < maxIteration; i++)
    {
      X[i].resize (n);
      Y[i].resize (n);
    }
  X[0] = x_old;

  while (err > tol && k <= maxIteration)
    {
      f (X[k], Y[k]);
      double err = 0.; // evaluate err
      for (unsigned int i = 0; i < Y.size (); i++)
	{
	  if (abs (Y[k][i]) >= err)
	    err = abs (Y[k][i]);
	}
      printf ("And_chen: iter=%d; err=%2.15E\n", k, err);
      if (err < tol)
	{
	  x_old = X[k];
	  printf (
	      "*****And_chen: Solved equation successfully!*****\nThe solution is:\n");
	  for (int i = 0; i < n; i++)
	    printf ("X[%d]=%2.15f\n", i, x_old[i]);
	  return 0;
	}

      restart: m = min (nm, k - k_restart);
      // Constuct matrix U and vector V
      Mat_DP U (m, m), V (m, 1);
      for (int i = 0; i < m; i++)
	{
	  for (int j = 0; j < m; j++)
	    {
	      U[i][j] = 0.;
	      for (int t = 0; t < n; t++)
		U[i][j] += (Y[k][t] - Y[k - i - 1][t])
		    * (Y[k][t] - Y[k - j - 1][t]);
	    }
	  V[i][0] = 0.;
	  for (int t = 0; t < n; t++)
	    V[i][0] += (Y[k][t] - Y[k - i - 1][t]) * Y[k][t];
	}

      int fail = 0;
      fail = NR::gaussj (U, V);

      if (fail == 1)
	{
	  printf ("And_chen: Singular Matrix detected And_chen restarted!\n");
	  k_restart = k;
	  lk = lmd;
	  goto restart;
	}
// update new X;
      for (int i = 0; i < n; i++)
	{
	  double cx = 0., cd = 0.;
	  for (int j = 0; j < m; j++)
	    {
	      cx += V[j][0] * (X[k - j - 1][i] - X[k][i]);
	      cd += V[j][0] * (Y[k - j - 1][i] - Y[k][i]);
	    }
	  X[k + 1][i] = X[k][i] + cx + (1 - lk) * (Y[k][i] + cd);
	}
      lk *= lmd;
      k++;
    }

  printf (
      "And_chen failed after %d iterations :(  Try to increase max iteration allowed\n",
      maxIteration);
  return 1;
}

