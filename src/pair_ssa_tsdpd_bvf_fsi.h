/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(ssa_tsdpd/bvf/fsi,PairSsaTsdpdBvfFsi)

#else

#ifndef LMP_PAIR_SSA_TSDPD_BVF_FSI_H
#define LMP_PAIR_SSA_TSDPD_BVF_FSI_H

#include "pair.h"

namespace LAMMPS_NS {

class PairSsaTsdpdBvfFsi : public Pair {
 public:
  PairSsaTsdpdBvfFsi(class LAMMPS *);
  virtual ~PairSsaTsdpdBvfFsi();
  void init_style();
  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  virtual double init_one(int, int);
  virtual double single(int, int, int, int, double, double, double, double &);
  class RanMars *random;

 protected:
  double *rho0, *soundspeed, *B;
  double **cut,**viscosity;
  double *temperature;
  double ***kappa;
  double ***kappaSSA;
  double **cutc;
  int first;
  unsigned int seed;
  double **rotationTensor;
  double **strainTensor;
  double **kroneckerTensor;
  double **transportTensor;
  double *G0;
  double ***kernelCorrectionTensor;
  void allocate();
};

}

#endif
#endif
