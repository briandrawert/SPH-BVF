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

#include <cmath>
#include <cstdlib>
#include <cstring>
#include "fix_dt_adaptive.h"
#include "atom.h"
#include "update.h"
#include "integrate.h"
#include "domain.h"
#include "lattice.h"
#include "force.h"
#include "pair.h"
#include "modify.h"
#include "fix.h"
#include "output.h"
#include "dump.h"
#include "comm.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace FixConst;

#define BIG 1.0e20

/* ---------------------------------------------------------------------- */

FixDtAdaptive::FixDtAdaptive(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 8) error->all(FLERR,"Illegal fix dt/adaptive command");

  // set time_depend, else elapsed time accumulation can be messed up

  time_depend = 1;
  scalar_flag = 1;
  global_freq = 1;
  extscalar = 0;
  extvector = 0;
  dynamic_group_allow = 1;

  nevery = force->inumeric(FLERR,arg[3]);
  if (nevery <= 0) error->all(FLERR,"Illegal fix dt/adaptive command");

  minbound = maxbound = 1;
  tmin = tmax = 0.0;
  if (strcmp(arg[4],"NULL") == 0) minbound = 0;
  else tmin = force->numeric(FLERR,arg[4]);
  if (strcmp(arg[5],"NULL") == 0) maxbound = 0;
  else tmax = force->numeric(FLERR,arg[5]);
  CFLmax = force->numeric(FLERR,arg[6]);
  dxAve = force->numeric(FLERR,arg[7]);

  if (minbound && tmin < 0.0) error->all(FLERR,"Illegal fix dt/adaptive command");
  if (maxbound && tmax < 0.0) error->all(FLERR,"Illegal fix dt/adaptive command");
  if (minbound && maxbound && tmin >= tmax) error->all(FLERR,"Illegal fix dt/adaptive command");
  if (CFLmax <= 0.0) error->all(FLERR,"Illegal fix dt/adaptive command");
  if (dxAve <= 0.0) error->all(FLERR,"Illegal fix dt/adaptive command");

  int scaleflag = 1;

  int iarg = 7;

  // setup scaling, based on xlattice parameter

  //if (scaleflag) xmax *= domain->lattice->xlattice;

  // initializations

  t_laststep = 0.0;
  laststep = update->ntimestep;
}

/* ---------------------------------------------------------------------- */

int FixDtAdaptive::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixDtAdaptive::init()
{
  // check for DCD or XTC dumps
  for (int i = 0; i < output->ndump; i++)
    if ((strcmp(output->dump[i]->style,"dcd") == 0 ||
        strcmp(output->dump[i]->style,"xtc") == 0) && comm->me == 0)
      error->warning(FLERR,"Dump dcd/xtc timestamp may be wrong with fix dt/adaptive");

  ftm2v = force->ftm2v;
  mvv2e = force->mvv2e;
  dt = update->dt;
}

/* ---------------------------------------------------------------------- */

void FixDtAdaptive::setup(int vflag)
{
  end_of_step();
}

/* ---------------------------------------------------------------------- */

void FixDtAdaptive::end_of_step()
{
  double dtv,dtf,dte,dtsq;
  double vsq,fsq,massinv;
  double delx,dely,delz,delr;

  double **v = atom->v;
  double **f = atom->f;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double dtmin = BIG;
  double maxVsq = 0.0;
  double Vsq = 0.0;
  double maxAllVsq=0.0;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      // Find maximum velocity for each processor
      Vsq = v[i][0]*v[i][0] + v[i][1]*v[i][1] + v[i][2]*v[i][2];
      maxVsq = MAX(maxVsq,Vsq);
    }
  // Find maximum velocity among all processors
  MPI_Allreduce(&maxVsq,&maxAllVsq,1,MPI_DOUBLE,MPI_MAX,world);

  // Compute dt using CFL
  dt = CFLmax*dxAve/sqrt(maxAllVsq);
  
  if (minbound) dt = MAX(dt,tmin);
  if (maxbound) dt = MIN(dt,tmax);

  // if timestep didn't change, just return
  // else adaptive update->dt and other classes that depend on it
  // rRESPA, pair style, fixes

  if (dt == update->dt) return;

  laststep = update->ntimestep;

  update->update_time();
  update->dt = dt;
  if (force->pair) force->pair->reset_dt();
  for (int i = 0; i < modify->nfix; i++) modify->fix[i]->reset_dt();
}

/* ---------------------------------------------------------------------- */

double FixDtAdaptive::compute_scalar()
{
  return (double) laststep;
}
