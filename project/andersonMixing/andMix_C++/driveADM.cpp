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
  out[0] = in[0] * in[1] * in[2] - 12.;
  out[1] = in[1] * in[1] + in[0] * in[0] - 8.;
  out[2] = in[1] + in[0] + in[2] - 5.;
}

int
main ()
{
  vector<double> x;
  x.push_back (1.);
  x.push_back (2.);
  x.push_back (3.);
  int fail = adm_chen (&myfun, x, 1e-15, 3000);

  if (~fail)
    for (unsigned int i=0;i<x.size();i++)
     printf("%2.15f\n",x[i]);

  return 0;
}
