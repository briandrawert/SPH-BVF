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
#include "compute_ssa_tsdpd_C_atom.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeSsaTsdpdCAtom::ComputeSsaTsdpdCAtom(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg != 4) error->all(FLERR,"Illegal compute ssa_tsdpd/C/atom command");

  peratom_flag = 1;
  size_peratom_cols = 0;
  nmax = 0;

  // Read argument (species number)
  species = force->inumeric(FLERR,arg[3]);

  CVector = NULL;

}

/* ---------------------------------------------------------------------- */

ComputeSsaTsdpdCAtom::~ComputeSsaTsdpdCAtom()
{
  memory->sfree(CVector);
}

/* ---------------------------------------------------------------------- */

void ComputeSsaTsdpdCAtom::init()
{

  int count = 0;
  for (int i = 0; i < modify->ncompute; i++)
    if (strcmp(modify->compute[i]->style,"CVector/atom") == 0) count++;
  if (count > 1 && comm->me == 0)
    error->warning(FLERR,"More than one compute CVector/atom");
}

/* ---------------------------------------------------------------------- */

void ComputeSsaTsdpdCAtom::compute_peratom()
{
  invoked_peratom = update->ntimestep;

  // grow CVector array if necessary

  if (atom->nmax > nmax) {
    memory->sfree(CVector);
    nmax = atom->nmax;
    CVector = (double *) memory->smalloc(nmax*sizeof(double),"atom:phiVector");
    vector_atom = CVector;
  }

  // compute kinetic energy for each atom in group

  double **C = atom->C;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
              CVector[i] = C[i][species];
      }
      else {
              CVector[i] = 0.0;
      }
    }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeSsaTsdpdCAtom::memory_usage()
{
  double bytes = nmax * sizeof(double);
  return bytes;
}
