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

#ifdef FIX_CLASS

FixStyle(dt/adaptive,FixDtAdaptive)

#else

#ifndef LMP_FIX_DT_ADAPTIVE_H
#define LMP_FIX_DT_ADAPTIVE_H

#include "fix.h"

namespace LAMMPS_NS {

class FixDtAdaptive : public Fix {
 public:
  FixDtAdaptive(class LAMMPS *, int, char **);
  ~FixDtAdaptive() {}
  int setmask();
  void init();
  void setup(int);
  void end_of_step();
  double compute_scalar();

 private:
  bigint laststep;
  int minbound,maxbound;
  double tmin,tmax,CFLmax,dxAve;
  double ftm2v,mvv2e;
  double dt,t_laststep;
  int respaflag;
};

}

#endif
#endif

