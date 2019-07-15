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

#include <string.h>
#include "compute_ssa_tsdpd_stress_atom.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeSsaTsdpdStressAtom::ComputeSsaTsdpdStressAtom(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg != 5) error->all(FLERR,"Illegal compute ssa_tsdpd/stress/atom command");

  peratom_flag = 1;
  size_peratom_cols = 0;
  nmax = 0;

  // Read argument (stress component number)
  // e.g., xx = 1,1; xy = 1,2, etc.
  
  component1 = force->inumeric(FLERR,arg[3]);
  component2 = force->inumeric(FLERR,arg[4]);

  stressVector = NULL;

}

/* ---------------------------------------------------------------------- */

ComputeSsaTsdpdStressAtom::~ComputeSsaTsdpdStressAtom()
{
  memory->sfree(stressVector);
}

/* ---------------------------------------------------------------------- */

void ComputeSsaTsdpdStressAtom::init()
{

  int count = 0;
  for (int i = 0; i < modify->ncompute; i++)
    if (strcmp(modify->compute[i]->style,"stressVector/atom") == 0) count++;
  if (count > 1 && comm->me == 0)
    error->warning(FLERR,"More than one compute stressVector/atom");
}

/* ---------------------------------------------------------------------- */

void ComputeSsaTsdpdStressAtom::compute_peratom()
{
  invoked_peratom = update->ntimestep;

  // grow stressVector array if necessary

  if (atom->nmax > nmax) {
    memory->sfree(stressVector);
    nmax = atom->nmax;
    stressVector = (double *) memory->smalloc(nmax*sizeof(double),"atom:stressVector");
    vector_atom = stressVector;
  }

  // compute kinetic energy for each atom in group

  double ***deviatoricTensor = atom->deviatoricTensor;
  double *Pnew = atom->Pnew;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
              if (component1 == component2) stressVector[i] = - Pnew[i] + deviatoricTensor[i][component1][component2];
              else stressVector[i] = deviatoricTensor[i][component1][component2];              
      }
      else {
              stressVector[i] = 0.0;
      }
    }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeSsaTsdpdStressAtom::memory_usage()
{
  double bytes = nmax * sizeof(double);
  return bytes;
}
