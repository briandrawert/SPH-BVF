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

#ifdef COMPUTE_CLASS

ComputeStyle(ssa_tsdpd/stress/atom,ComputeSsaTsdpdStressAtom)

#else

#ifndef LMP_COMPUTE_SSA_TSDPD_STRESS_ATOM_H
#define LMP_COMPUTE_SSA_TSDPD_STRESS_ATOM_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeSsaTsdpdStressAtom : public Compute {
 public:
  ComputeSsaTsdpdStressAtom(class LAMMPS *, int, char **);
  ~ComputeSsaTsdpdStressAtom();
  void init();
  void compute_peratom();
  double memory_usage();

 private:
  int nmax;
  int component1,component2;
  double *stressVector;
};

}

#endif
#endif
