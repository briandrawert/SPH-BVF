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

#ifdef ATOM_CLASS

AtomStyle(ssa_tsdpd/atomic,AtomVecSsaTsdpdAtomic)

#else

#ifndef LMP_ATOM_VEC_SSA_TSDPD_ATOMIC_H
#define LMP_ATOM_VEC_SSA_TSDPD_ATOMIC_H

#include "atom_vec.h"

namespace LAMMPS_NS {

class AtomVecSsaTsdpdAtomic : public AtomVec {
 public:
  AtomVecSsaTsdpdAtomic(class LAMMPS *);
  ~AtomVecSsaTsdpdAtomic() {}
  void process_args(int, char **);
  void grow(int);
  void grow_reset();
  void copy(int, int, int);
  void force_clear(int, size_t);
  int pack_comm(int, int *, double *, int, int *);
  int pack_comm_vel(int, int *, double *, int, int *);
  void unpack_comm(int, int, double *);
  void unpack_comm_vel(int, int, double *);
  int pack_reverse(int, int, double *);
  void unpack_reverse(int, int *, double *);
  int pack_border(int, int *, double *, int, int *);
  int pack_border_vel(int, int *, double *, int, int *);
  void unpack_border(int, int, double *);
  void unpack_border_vel(int, int, double *);
  int pack_exchange(int, double *);
  int unpack_exchange(double *);
  int size_restart();
  int pack_restart(int, double *);
  int unpack_restart(double *);
  void create_atom(int, double *);
  void data_atom(double *, imageint, char **);
  void pack_data(double **);
  void write_data(FILE *, int, double **);
  int property_atom(char *);
  void pack_property_atom(int, double *, int, int);
  bigint memory_usage();

 private:
  tagint *tag;
  int *type,*mask;
  imageint *image;
  tagint *molecule;
  double **x,**v,**f;
  double *rho, *drho, *e, *de, *cv;
  double **vest; // estimated velocity during force computation
  double **C, **Q; //added tDPD/tSDPD variables
  int **Cd, **Qd; //added tDPD/tSDPD variables (SSA)
  double **ssa_rxn_propensity, ***d_ssa_rxn_prop_d_c;  // SSA reaction propensities, SSA reaction jacobian
  int ***ssa_stoich_matrix; // SSA reaction species change matrix
  double *dfsp_D_matrix,*dfsp_D_diag,*dfsp_Diffusion_coeff,*dfsp_a_i;
  int *solid_tag;
  int *fixed_tag;
  double *phi;
  double *number_density;
  double **nw;
  double **v_weighted_solid, **a_weighted_solid;
  double ***deviatoricTensor, ***ddeviatoricTensor;
  double ***artificialStressTensor;
  //double ***kernelCorrectionTensor;
  double **ddx, **ddv;
  double *Pold, *Pnew, *Aaux, *Baux, *APaux;
  double **fP;
  double *rhoI;
  double *rhoAux1,*rhoAux2,*rhoAux3;
};

}

#endif
#endif
