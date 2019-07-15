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
#include "compute_ssa_tsdpd_solid_tag_atom.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeSsaTsdpdSolidTagAtom::ComputeSsaTsdpdSolidTagAtom(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg != 3) error->all(FLERR,"Illegal compute ssa_tsdpd/phi/atom command");

  peratom_flag = 1;
  size_peratom_cols = 0;

  nmax = 0;
  solidTagVector = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeSsaTsdpdSolidTagAtom::~ComputeSsaTsdpdSolidTagAtom()
{
  memory->sfree(solidTagVector);
}

/* ---------------------------------------------------------------------- */

void ComputeSsaTsdpdSolidTagAtom::init()
{

  int count = 0;
  for (int i = 0; i < modify->ncompute; i++)
    if (strcmp(modify->compute[i]->style,"solidTagVector/atom") == 0) count++;
  if (count > 1 && comm->me == 0)
    error->warning(FLERR,"More than one compute solidTagVector/atom");
}

/* ---------------------------------------------------------------------- */

void ComputeSsaTsdpdSolidTagAtom::compute_peratom()
{
  invoked_peratom = update->ntimestep;

  // grow solidTagVector array if necessary

  if (atom->nmax > nmax) {
    memory->sfree(solidTagVector);
    nmax = atom->nmax;
    solidTagVector = (double *) memory->smalloc(nmax*sizeof(double),"atom:solidTagVector");
    vector_atom = solidTagVector;
  }

  // compute kinetic energy for each atom in group

  int *solid_tag = atom->solid_tag;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
              solidTagVector[i] = (double) solid_tag[i];
      }
      else {
              solidTagVector[i] = 0.0;
      }
    }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeSsaTsdpdSolidTagAtom::memory_usage()
{
  double bytes = nmax * sizeof(double);
  return bytes;
}
