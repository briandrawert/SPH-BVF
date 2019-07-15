/* ----------------------------------------------------------------------
 LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
 http://lammps.sandia.gov, Sandia National Laboratories
 Steve Plimpton, sjplimp@sandia.gov

 Copyright (2003) Sandia Corporation.  Under the terms of Contract
 DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
 certain rights in this software.  This software is distributed under
 the GNU General Public License.

 See the README file in the top-level LAMMPS directory.
 ------------------------------------------------------------------------- */

#include <stdio.h>
#include <string.h>
#include "fix_ssa_tsdpd_stationary.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "integrate.h"
#include "respa.h"
#include "memory.h"
#include "error.h"
#include "pair.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixSsaTsdpdStationary::FixSsaTsdpdStationary(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg) {

  if ((atom->e_flag != 1) || (atom->rho_flag != 1))
    error->all(FLERR,
        "fix ssa_tsdpd/stationary command requires atom_style with both energy and density, e.g. ssa_tsdpd");

  if (narg != 3)
    error->all(FLERR,"Illegal number of arguments for fix ssa_tsdpd/stationary command");

  time_integrate = 0;
  
  seed = comm->nprocs + comm->me + atom->nlocal;
  random = new RanMars (lmp, seed);
}

/* ---------------------------------------------------------------------- */

int FixSsaTsdpdStationary::setmask() {
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixSsaTsdpdStationary::init() {
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
}

/* ----------------------------------------------------------------------
 allow for both per-type and per-atom mass
 ------------------------------------------------------------------------- */

void FixSsaTsdpdStationary::initial_integrate(int vflag) {

  double *rho = atom->rho;
  double *drho = atom->drho;
  double *e = atom->e;
  double *de = atom->de;
  
  double **C = atom->C;
  double **Q = atom->Q;

  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int i;

  if (igroup == atom->firstgroup)
    nlocal = atom->nfirst;

  for (i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
  //    e[i] += dtf * de[i]; // half-step update of particle internal energy
      rho[i] += dtf * drho[i]; // ... and density
      for (int k = 0; k < atom->num_sdpd_species; k++){ // ...and concentrations
        C[i][k] += Q[i][k] *dtf;
        C[i][k] = C[i][k] > 0 ? C[i][k] : 0.0;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixSsaTsdpdStationary::final_integrate() {

//  printf("fix_ssa_tsdpd_stationary.cpp at t = %d", update->ntimestep);
  double *e = atom->e;
  double *de = atom->de;
  double *rho = atom->rho;
  double *drho = atom->drho;
 
  double **C = atom->C;
  double **Q = atom->Q;

  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int nmax = atom->nmax;

  if (igroup == atom->firstgroup)
    nlocal = atom->nfirst;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
//    e[i] += dtf * de[i];
      rho[i] += dtf * drho[i];
      for (int k = 0; k < atom->num_sdpd_species; k++){
        C[i][k] += Q[i][k] *dtf;
        C[i][k] = C[i][k] > 0 ? C[i][k] : 0.0;
      }
    }  
  }
}

/* ---------------------------------------------------------------------- */

void FixSsaTsdpdStationary::reset_dt() {
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
}
