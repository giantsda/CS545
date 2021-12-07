/*
 * driveADM.c
 *
 *  Created on: Oct 6, 2018
 *      Author: chen
 */

#include <vector>  // std::vector
#include "nr.h"  //gaussj
#include <stdio.h> // printf()
#include <cmath>        // std::abs
using namespace std;
int
adm_chen (void
(*f) (vector<double>& in, vector<double>& out),
	  vector<double>& x_old, double tol, int maxIteration);

void
myfun (vector<double>& in, vector<double>& out)
{
  out[0] = 0.3*in[0] + 0.1*in[1] + 0.12*in[2] - 0.3;
  out[1] = 0.13*in[0] + 0.21*in[1] + 0.212*in[2] - 0.13;
  out[2] = 0.01*in[0] + 0.22*in[1] + 0.122*in[2] - 0.33;
  out[3] = 0.212*in[0] + 0.7*in[1] + 0.132*in[2] - 0.223;
  out[4] = 0.23*in[0] + 0.11*in[1] + 0.162*in[2] - 0.334;
  out[5] = 0.33*in[0] + 0.21*in[1] + 0.112*in[2] - 0.2;
  out[6] = 0.43*in[0] + 0.31*in[1] + 0.122*in[2] - 0.1;
  out[7] = 0.53*in[0] + 0.41*in[1] + 0.132*in[2] - 0.222;
  out[8] = 0.73*in[0] + 0.51*in[1] + 0.162*in[2] - 0.6;
}

int
main ()
{
  vector<double> x;
  x.push_back (1.);
  x.push_back (2.);
  x.push_back (3.);
  x.push_back (3.);
  x.push_back (3.);
  x.push_back (3.);
  x.push_back (3.);
  x.push_back (3.);
  
  int fail = adm_chen (&myfun, x, 1e-15, 3000);

  if (~fail)
    for (unsigned int i=0;i<x.size();i++)
     printf("%2.15f\n",x[i]);

  return 0;
}
