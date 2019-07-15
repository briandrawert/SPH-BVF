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
#include <stdlib.h>
#include "atom_vec_ssa_tsdpd_atomic.h"
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "modify.h"
#include "fix.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

AtomVecSsaTsdpdAtomic::AtomVecSsaTsdpdAtomic(LAMMPS *lmp) : AtomVec(lmp)
{
  molecular = 0;
  mass_type = 1;

  comm_x_only = 0;        // force communication of vest (as well as x) forward
  comm_f_only = 0;        // communicate de and drho in reverse direction
  size_forward = 22;      // 3 + rho + e + vest[3], that means we may only communicate 5 in hybrid (was 8 before)
  size_reverse = 51;      // 3 + drho + de
  size_border = 27;       // 6 + rho + e + vest[3] + cv
  size_velocity = 3;
  size_data_atom = 8;     // total number of columns in atom input file
  size_data_vel = 3;    
  xcol_data = 5;          // column where the x,y,z coords start in the input file

  atom->e_flag = 1;
  atom->rho_flag = 1;
  atom->cv_flag = 1;
  atom->vest_flag = 1;
  atom->sdpd_flag = 1;
  forceclearflag = 1;
}


/* ----------------------------------------------------------------------
   process additional args
   set size_forward and size_border to max sizes
------------------------------------------------------------------------- */

void AtomVecSsaTsdpdAtomic::process_args(int narg, char **arg)
{
  
  if (narg < 1) error->all(FLERR,"Must provide NUMBER OF DETERMINISTIC SPECIES (int).");
  if (narg > 3) error->all(FLERR,"Must provide NUMBER OF STOCHASTIC SPECIES (int) and NUMBER OF STOCHASTIC REACTIONS (int).");

   
  if (narg ==1) {
    atom->num_sdpd_species = atoi(arg[0]);
    if (comm->me == 0) printf("num_sdpd_species = %d \n",atom->num_sdpd_species);

    size_forward += atom->num_sdpd_species;
    size_reverse += atom->num_sdpd_species;
    size_border  += atom->num_sdpd_species;
    //size_data_atom += atom->num_sdpd_species;
  }

  else if (narg < 4 && narg > 2) {

    if (atoi(arg[0]) >= 0) {
      atom->num_sdpd_species = atoi(arg[0]);
      if (comm->me == 0) printf("num_sdpd_species = %d \n",atom->num_sdpd_species);
    }
    else error->all(FLERR,"Number of SDPD species must be greater than or equal to 0");

    if (atoi(arg[1]) >= 0) {
      atom->num_ssa_species = atoi(arg[1]);
      if (comm->me == 0)  printf("num_ssa_species = %d \n",atom->num_ssa_species);
    }
    else error->all(FLERR,"Number of SSA species must be greater than or equal to 0");


    if (atoi(arg[2]) >= 0) {
      atom->num_ssa_reactions = atoi(arg[2]);
      if (comm->me == 0) printf("num_ssa_reactions = %d \n",atom->num_ssa_reactions);
    }
    else error->all(FLERR,"Number of SSA reactions must be greater than or equal to 0");


    if (comm->nprocs > 1 && atoi(arg[1]) > 0) error->all(FLERR,"Invalid MPI run. Currently, stochastic simulations with SSA only works in serial. Please change the number of MPI processors to 1, or set the number of SSA species and reactions to 0.");

    if (atom->num_ssa_species > 0) atom->ssa_diffusion_flag = 1;

    if (atom->num_ssa_reactions > 0) atom->ssa_reaction_flag = 1;

    int size_extra = 2 * atom->num_ssa_reactions * atom->num_ssa_species + 2 + atom->num_ssa_reactions + atom->nmax + atom->num_ssa_species;
    size_forward += atom->num_sdpd_species + size_extra;
    size_reverse += atom->num_sdpd_species + atom->num_ssa_species;
    size_border  += atom->num_sdpd_species + size_extra;
  }
}

/* ----------------------------------------------------------------------
   grow atom arrays
   n = 0 grows arrays by a chunk
   n > 0 allocates arrays to size n
   ------------------------------------------------------------------------- */

void AtomVecSsaTsdpdAtomic::grow(int n)
{

//  printf("in AtomVecSsaTsdpdAtomic::grow\n");

  int num_sdpd_species = atom->num_sdpd_species;
 
  if (n == 0) grow_nmax();
  else nmax = n;
  atom->nmax = nmax;
  if (nmax < 0 || nmax > MAXSMALLINT)
    error->one(FLERR,"Per-processor system is too big");

  tag = memory->grow(atom->tag, nmax, "atom:tag");
  type = memory->grow(atom->type, nmax, "atom:type");
  mask = memory->grow(atom->mask, nmax, "atom:mask");
  image = memory->grow(atom->image, nmax, "atom:image");
  x = memory->grow(atom->x, nmax, 3, "atom:x");
  v = memory->grow(atom->v, nmax, 3, "atom:v");
  f = memory->grow(atom->f, nmax*comm->nthreads, 3, "atom:f");

  rho = memory->grow(atom->rho, nmax, "atom:rho");
  drho = memory->grow(atom->drho, nmax*comm->nthreads, "atom:drho");
  e = memory->grow(atom->e, nmax, "atom:e");
  de = memory->grow(atom->de, nmax*comm->nthreads, "atom:de");
  vest = memory->grow(atom->vest, nmax, 3, "atom:vest");
  cv = memory->grow(atom->cv, nmax, "atom:cv");
 
  C = memory->grow(atom->C,nmax,atom->num_sdpd_species,"atom:C");
  Q = memory->grow(atom->Q,nmax*comm->nthreads,atom->num_sdpd_species,"atom:Q");

  int num_ssa_species = atom->num_ssa_species;
  int num_ssa_reactions = atom->num_ssa_reactions;
 
  if (atom->ssa_diffusion_flag == 1) {
    Cd = memory->grow(atom->Cd,nmax,num_ssa_species,"atom:Cd");
    Qd = memory->grow(atom->Qd,nmax*comm->nthreads,num_ssa_species,"atom:Qd");
    dfsp_D_matrix = memory->grow(atom->dfsp_D_matrix,nmax*nmax,"atom:dfsp_D_matrix");
    dfsp_D_diag = memory->grow(atom->dfsp_D_diag,nmax,"atom:dfsp_D_diag");
    //dfsp_Diffusion_coeff = memory->grow(atom->dfsp_Diffusion_coeff,nmax*nmax*num_ssa_species,"atom:dfsp_Diffusion_coeff");
    dfsp_a_i = memory->grow(atom->dfsp_a_i, nmax,"atom:dfsp_a_i");
  }

  if (atom->ssa_reaction_flag == 1) {
    ssa_rxn_propensity = memory->grow(atom->ssa_rxn_propensity,nmax,num_ssa_reactions,"atom:ssa_rxn_propensity");
    d_ssa_rxn_prop_d_c = memory->grow(atom->d_ssa_rxn_prop_d_c,nmax,num_ssa_reactions,num_ssa_species,"atom:d_ssa_rxn_prop_d_c");
    ssa_stoich_matrix = memory->grow(atom->ssa_stoich_matrix,nmax,num_ssa_reactions,num_ssa_species,"atom:ssa_stoich_matrix");
  }

  solid_tag = memory->grow(atom->solid_tag, nmax, "atom:solid_tag");
  fixed_tag = memory->grow(atom->fixed_tag, nmax, "atom:fixed_tag");
  phi = memory->grow(atom->phi, nmax*comm->nthreads, "atom:phi");
  number_density = memory->grow(atom->number_density, nmax*comm->nthreads, "atom:number_density");
  nw = memory->grow(atom->nw, nmax*comm->nthreads, 3, "atom:nw");
  v_weighted_solid = memory->grow(atom->v_weighted_solid, nmax*comm->nthreads, 3, "atom:v_weighted_solid");
  a_weighted_solid = memory->grow(atom->a_weighted_solid, nmax*comm->nthreads, 3, "atom:a_weighted_solid");
  deviatoricTensor = memory->grow(atom->deviatoricTensor, nmax, 3, 3, "atom:deviatoricTensor");
  ddeviatoricTensor = memory->grow(atom->ddeviatoricTensor, nmax*comm->nthreads, 3, 3, "atom:ddeviatoricTensor");
  artificialStressTensor = memory->grow(atom->artificialStressTensor, nmax*comm->nthreads, 3, 3, "atom:artificialStressTensor");
  //kernelCorrectionTensor = memory->grow(atom->kernelCorrectionTensor, nmax*comm->nthreads, 3, 3, "atom:kernelCorrectionTensor");

  ddx = memory->grow(atom->ddx, nmax*comm->nthreads, 3, "atom:ddx");
  ddv = memory->grow(atom->ddv, nmax*comm->nthreads, 3, "atom:ddv");
  
  Pold  = memory->grow(atom->Pold, nmax*comm->nthreads, "atom:Pold");
  Pnew  = memory->grow(atom->Pnew, nmax*comm->nthreads, "atom:Pnew");
  Aaux  = memory->grow(atom->Aaux, nmax*comm->nthreads, "atom:Aaux");
  Baux  = memory->grow(atom->Baux, nmax*comm->nthreads, "atom:Baux");
  APaux = memory->grow(atom->APaux, nmax*comm->nthreads, "atom:APaux");
  fP    = memory->grow(atom->fP, nmax*comm->nthreads,3, "atom:fP");
  rhoI  = memory->grow(atom->rhoI, nmax, "atom:rhoI");
  rhoAux1  = memory->grow(atom->rhoAux1, nmax, "atom:rhoAux1");
  rhoAux2  = memory->grow(atom->rhoAux2, nmax, "atom:rhoAux2");
  rhoAux3  = memory->grow(atom->rhoAux3, nmax, "atom:rhoAux3");

  if (atom->nextra_grow)
    for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
      modify->fix[atom->extra_grow[iextra]]->grow_arrays(nmax);

}

/* ----------------------------------------------------------------------
   reset local array ptrs
   ------------------------------------------------------------------------- */

void AtomVecSsaTsdpdAtomic::grow_reset() {

//  printf("in AtomVecSsaTsdpdAtomic::reset\n");

  tag = atom->tag;
  type = atom->type;
  mask = atom->mask;
  image = atom->image;
  x = atom->x;
  v = atom->v;
  f = atom->f;
  /*
  molecule = atom->molecule;
  nspecial = atom->nspecial; special = atom->special;
  num_bond = atom->num_bond; bond_type = atom->bond_type;
  bond_atom = atom->bond_atom;
  num_angle = atom->num_angle; angle_type = atom->angle_type;
  angle_atom1 = atom->angle_atom1; angle_atom2 = atom->angle_atom2;
  angle_atom3 = atom->angle_atom3;
  */
  rho = atom->rho;
  drho = atom->drho;
  e = atom->e;
  de = atom->de;
  vest = atom->vest;
  cv = atom->cv;
  C = atom->C; 
  Q = atom->Q; 

  if (atom->ssa_diffusion_flag == 1) {
    Cd = atom->Cd;
    Qd = atom->Qd;
    dfsp_D_matrix = atom->dfsp_D_matrix;
    dfsp_D_diag = atom->dfsp_D_diag;
    //dfsp_Diffusion_coeff = atom->dfsp_Diffusion_coeff;
    dfsp_a_i = atom->dfsp_a_i;
  }

  if (atom->ssa_reaction_flag == 1) {
    ssa_rxn_propensity = atom->ssa_rxn_propensity;
    d_ssa_rxn_prop_d_c = atom->d_ssa_rxn_prop_d_c;
    ssa_stoich_matrix = atom->ssa_stoich_matrix;
  }

  solid_tag = atom->solid_tag;
  fixed_tag = atom->fixed_tag;
  phi = atom->phi;
  number_density = atom->number_density;
  nw = atom->nw;
  v_weighted_solid = atom->v_weighted_solid;
  a_weighted_solid = atom->a_weighted_solid;
  deviatoricTensor = atom->deviatoricTensor;
  ddeviatoricTensor = atom->ddeviatoricTensor;
  artificialStressTensor = atom->artificialStressTensor;
  //kernelCorrectionTensor = atom->kernelCorrectionTensor;
  ddx = atom->ddx;
  ddv = atom->ddv;
  
  Pold  = atom->Pold;
  Pnew  = atom->Pnew;
  Aaux  = atom->Aaux;
  Baux  = atom->Baux;
  APaux = atom->APaux;
  fP = atom->fP;
  rhoI = atom->rhoI;
  rhoAux1 = atom->rhoAux1;
  rhoAux2 = atom->rhoAux2;
  rhoAux3 = atom->rhoAux3;
}

/* ----------------------------------------------------------------------
   copy atom I info to atom J
 ---------------------------------------------------------------------- */

void AtomVecSsaTsdpdAtomic::copy(int i, int j, int delflag) 
{

  int k, r;
    
  tag[j] = tag[i];
  type[j] = type[i];
  mask[j] = mask[i];
  image[j] = image[i];
  x[j][0] = x[i][0];
  x[j][1] = x[i][1];
  x[j][2] = x[i][2];
  v[j][0] = v[i][0];
  v[j][1] = v[i][1];
  v[j][2] = v[i][2];

  /*
  molecule[j] = molecule[i];
  num_bond[j] = num_bond[i];
  for (k = 0; k < num_bond[j]; k++) {
    bond_type[j][k] = bond_type[i][k];
    bond_atom[j][k] = bond_atom[i][k];
  }
  num_angle[j] = num_angle[i];
  for (k = 0; k < num_angle[j]; k++) {
    angle_type[j][k] = angle_type[i][k];
    angle_atom1[j][k] = angle_atom1[i][k];
    angle_atom2[j][k] = angle_atom2[i][k];
    angle_atom3[j][k] = angle_atom3[i][k];
  }
  nspecial[j][0] = nspecial[i][0];
  nspecial[j][1] = nspecial[i][1];
  nspecial[j][2] = nspecial[i][2];
  for (k = 0; k < nspecial[j][2]; k++) special[j][k] = special[i][k];
   */
   
  rho[j] = rho[i];
  drho[j] = drho[i];
  e[j] = e[i];
  de[j] = de[i];
  cv[j] = cv[i];
  vest[j][0] = vest[i][0];
  vest[j][1] = vest[i][1];
  vest[j][2] = vest[i][2];
  for (k = 0; k < atom->num_sdpd_species; k++)  C[j][k] = C[i][k];


  if (atom->ssa_diffusion_flag == 1) {
    for (k = 0; k < atom->num_ssa_species; k++) Cd[j][k] = Cd[i][k];
    dfsp_D_matrix[j] = dfsp_D_matrix[i];
    dfsp_D_diag[j] = dfsp_D_diag[i];
    //dfsp_Diffusion_coeff[j] = dfsp_Diffusion_coeff[i];
    dfsp_a_i[j] = dfsp_a_i[i];
  }

  if (atom->ssa_reaction_flag == 1) {
    for (r = 0; r < atom->num_ssa_reactions; r++) ssa_rxn_propensity[j][r] = ssa_rxn_propensity[i][r];
    for (r = 0; r < atom->num_ssa_reactions; r++)
      for (k = 0; k < atom->num_ssa_species; k++)
        d_ssa_rxn_prop_d_c[j][r][k] = d_ssa_rxn_prop_d_c[i][r][k];
    for (r = 0; r < atom->num_ssa_reactions; r++)
      for (k = 0; k < atom->num_ssa_species; k++)
        ssa_stoich_matrix[j][r][k] = ssa_stoich_matrix[i][r][k];
  }

  solid_tag[j] = solid_tag[i];
  fixed_tag[j] = fixed_tag[i];
  phi[j] = phi[i]; 
  number_density[j] = number_density[i]; 
  nw[j][0] = nw[i][0]; 
  nw[j][1] = nw[i][1]; 
  nw[j][2] = nw[i][2]; 
  v_weighted_solid[j][0] = v_weighted_solid[i][0]; 
  v_weighted_solid[j][1] = v_weighted_solid[i][1]; 
  v_weighted_solid[j][2] = v_weighted_solid[i][2]; 
  a_weighted_solid[j][0] = a_weighted_solid[i][0]; 
  a_weighted_solid[j][1] = a_weighted_solid[i][1]; 
  a_weighted_solid[j][2] = a_weighted_solid[i][2]; 

  for (int k = 0; k < 3; k++) {
    for (int r= 0; r < 3; r++){
      deviatoricTensor[j][k][r] = deviatoricTensor[i][k][r]; 
      ddeviatoricTensor[j][k][r] = ddeviatoricTensor[i][k][r]; 
      artificialStressTensor[j][k][r] = artificialStressTensor[i][k][r]; 
      //kernelCorrectionTensor[j][k][r] = kernelCorrectionTensor[i][k][r]; 
    }
  }
  
  ddx[j][0] = ddx[i][0];
  ddx[j][1] = ddx[i][1];
  ddx[j][2] = ddx[i][2];
  ddv[j][0] = ddv[i][0];
  ddv[j][1] = ddv[i][1];
  ddv[j][2] = ddv[i][2];
  
  Pold[j] = Pold[i];
  Pnew[j] = Pnew[i];
  Aaux[j] = Aaux[i];
  Baux[j] = Baux[i];
  APaux[j] = APaux[i];
  fP[j][0] = fP[i][0]; 
  fP[j][1] = fP[i][1]; 
  fP[j][2] = fP[i][2]; 
  rhoI[j] = rhoI[i];
  rhoAux1[j] = rhoAux1[i];
  rhoAux2[j] = rhoAux2[i];
  rhoAux3[j] = rhoAux3[i];

  if (atom->nextra_grow)
    for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
      modify->fix[atom->extra_grow[iextra]]->copy_arrays(i, j,delflag);

}

/* ---------------------------------------------------------------------- */

void AtomVecSsaTsdpdAtomic::force_clear(int n, size_t nbytes)
{

//  printf("in AtomVecSsaTsdpdAtomic::force_clear\n");

  memset(&de[n],0,nbytes);
  memset(&drho[n],0,nbytes);
  memset(&Q[n][0],0,atom->num_sdpd_species*nbytes);
  memset(&ddeviatoricTensor[n][0][0],0,9*nbytes);
  memset(&artificialStressTensor[n][0][0],0,9*nbytes);
  //memset(&kernelCorrectionTensor[n][0][0],0,9*nbytes);
  memset(&phi[n],0,nbytes);
  memset(&number_density[n],0,nbytes);
  memset(&nw[n][0],0,3*nbytes);
  memset(&v_weighted_solid[n][0],0,3*nbytes);
  memset(&a_weighted_solid[n][0],0,3*nbytes);
  memset(&ddx[n][0],0,3*nbytes);
  memset(&ddv[n][0],0,3*nbytes);

  memset(&Pold[n],0,nbytes);
  memset(&Pnew[n],0,nbytes);
  memset(&Aaux[n],0,nbytes);
  memset(&Baux[n],0,nbytes);
  memset(&APaux[n],0,nbytes);
  memset(&fP[n][0],0,3*nbytes);
  memset(&rhoAux1[n],0,nbytes);
  memset(&rhoAux2[n],0,nbytes);
  memset(&rhoAux3[n],0,nbytes);

  if (atom->ssa_diffusion_flag == 1) memset(&Qd[n][0],0,atom->num_ssa_species*nbytes);
    
}

/* ---------------------------------------------------------------------- */

int AtomVecSsaTsdpdAtomic::pack_comm(int n, int *list, double *buf, int pbc_flag,
                           int *pbc) {

//  printf("in AtomVecSsaTsdpdAtomic::pack_comm\n");

  int i, j, m, r, k;
  double dx, dy, dz;

  m = 0;
  if (pbc_flag == 0) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x[j][0];
      buf[m++] = x[j][1];
      buf[m++] = x[j][2];
      buf[m++] = v[j][0]; //
      buf[m++] = v[j][1]; //
      buf[m++] = v[j][2]; //
      buf[m++] = rho[j];
      buf[m++] = e[j];
      buf[m++] = vest[j][0];
      buf[m++] = vest[j][1];
      buf[m++] = vest[j][2];
      for (int k = 0; k < atom->num_sdpd_species; k++)  buf[m++] = C[j][k];

      if (atom->ssa_diffusion_flag == 1) {
        for (int k = 0; k < atom->num_ssa_species; k++)  buf[m++] = (double) Cd[j][k];
        buf[m++] = dfsp_D_matrix[j];
        buf[m++] = dfsp_D_diag[j];
        //buf[m++] = dfsp_Diffusion_coeff[j];
        buf[m++] = dfsp_a_i[j];
      }

      if (atom->ssa_reaction_flag == 1){
        for (int r = 0; r < atom->num_ssa_reactions; r++)  buf[m++] = ssa_rxn_propensity[j][r];
        for (int r = 0; r < atom->num_ssa_reactions; r++)
          for (int k = 0; k < atom->num_ssa_species; k++)
            buf[m++] = d_ssa_rxn_prop_d_c[j][r][k];
        for (int r = 0; r < atom->num_ssa_reactions; r++)
          for (int k = 0; k < atom->num_ssa_species; k++)
            buf[m++] = ssa_stoich_matrix[j][r][k];
      }
      
      //buf[m++] = phi[j];
      //buf[m++] = nw[j][0];
      //buf[m++] = nw[j][1];
      //buf[m++] = nw[j][2];
      //buf[m++] = v_weighted_solid[j][0];
      //buf[m++] = v_weighted_solid[j][1];
      //buf[m++] = v_weighted_solid[j][2];
      //buf[m++] = a_weighted_solid[j][0];
      //buf[m++] = a_weighted_solid[j][1];
      //buf[m++] = a_weighted_solid[j][2];

     for (int k = 0; k < 3; k++)
       for (int r = 0; r < 3; r++)
         buf[m++] = deviatoricTensor[j][k][r];

      buf[m++] = rhoI[j];
    }
  } else {
    if (domain->triclinic == 0) {
      dx = pbc[0] * domain->xprd;
      dy = pbc[1] * domain->yprd;
      dz = pbc[2] * domain->zprd;
    } else {
      dx = pbc[0] * domain->xprd + pbc[5] * domain->xy + pbc[4] * domain->xz;
      dy = pbc[1] * domain->yprd + pbc[3] * domain->yz;
      dz = pbc[2] * domain->zprd;
    }
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x[j][0] + dx;
      buf[m++] = x[j][1] + dy;
      buf[m++] = x[j][2] + dz;
      buf[m++] = v[j][0]; //
      buf[m++] = v[j][1]; //
      buf[m++] = v[j][2]; //
      buf[m++] = rho[j];
      buf[m++] = e[j];
      buf[m++] = vest[j][0];
      buf[m++] = vest[j][1];
      buf[m++] = vest[j][2];
      for (int k = 0; k < atom->num_sdpd_species; k++)  buf[m++] = C[j][k];

      if (atom->ssa_diffusion_flag == 1) {
        for (int k = 0; k < atom->num_ssa_species; k++)  buf[m++] = (double) Cd[j][k]; 
          buf[m++] = dfsp_D_matrix[j];
          buf[m++] = dfsp_D_diag[j];
          //buf[m++] = dfsp_Diffusion_coeff[j];
          buf[m++] = dfsp_a_i[j];
      }

      if (atom->ssa_reaction_flag == 1) {
        for (int r = 0; r < atom->num_ssa_reactions; r++)  buf[m++] = ssa_rxn_propensity[j][r];
        for (int r = 0; r < atom->num_ssa_reactions; r++)
          for (int k = 0; k < atom->num_ssa_species; k++)
            buf[m++] = d_ssa_rxn_prop_d_c[j][r][k];
        for (int r = 0; r < atom->num_ssa_reactions; r++)
          for (int k = 0; k < atom->num_ssa_species; k++)
            buf[m++] = ssa_stoich_matrix[j][r][k];
      }

      //buf[m++] = phi[j];
      //buf[m++] = nw[j][0];
      //buf[m++] = nw[j][1];
      //buf[m++] = nw[j][2];
      //buf[m++] = v_weighted_solid[j][0];
      //buf[m++] = v_weighted_solid[j][1];
      //buf[m++] = v_weighted_solid[j][2];
      //buf[m++] = a_weighted_solid[j][0];
      //buf[m++] = a_weighted_solid[j][1];
      //buf[m++] = a_weighted_solid[j][2];
      
      for (int k = 0; k < 3; k++)
        for (int r = 0; r < 3; r++)
          buf[m++] = deviatoricTensor[j][k][r];

      buf[m++] = rhoI[j];
      
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecSsaTsdpdAtomic::pack_comm_vel(int n, int *list, double *buf, int pbc_flag,
                               int *pbc) {

//  printf("in AtomVecSsaTsdpdAtomic::pack_comm_vel\n");

  int i, j, m, r, k;
  double dx, dy, dz;

  m = 0;
  if (pbc_flag == 0) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x[j][0];
      buf[m++] = x[j][1];
      buf[m++] = x[j][2];
      buf[m++] = v[j][0];
      buf[m++] = v[j][1];
      buf[m++] = v[j][2];
      buf[m++] = rho[j];
      buf[m++] = e[j];
      buf[m++] = vest[j][0];
      buf[m++] = vest[j][1];
      buf[m++] = vest[j][2];
      for (int k = 0; k < atom->num_sdpd_species; k++)  buf[m++] = C[j][k];

      if (atom->ssa_diffusion_flag == 1) {
        for (int k = 0; k < atom->num_ssa_species; k++)  buf[m++] = (double) Cd[j][k];
        buf[m++] = dfsp_D_matrix[j];
        buf[m++] = dfsp_D_diag[j];
        //buf[m++] = dfsp_Diffusion_coeff[j];
        buf[m++] = dfsp_a_i[j];      
      }

      if (atom->ssa_reaction_flag == 1) {
        for (int r = 0; r < atom->num_ssa_reactions; r++)  buf[m++] = ssa_rxn_propensity[j][r];
        for (int r = 0; r < atom->num_ssa_reactions; r++)
          for (int k = 0; k < atom->num_ssa_species; k++)
            buf[m++] = d_ssa_rxn_prop_d_c[j][r][k];
        for (int r = 0; r < atom->num_ssa_reactions; r++)
          for (int k = 0; k < atom->num_ssa_species; k++)
            buf[m++] = ssa_stoich_matrix[j][r][k];
      }

      //buf[m++] = phi[j];
      //buf[m++] = nw[j][0];
      //buf[m++] = nw[j][1];
      //buf[m++] = nw[j][2];
      //buf[m++] = v_weighted_solid[j][0];
      //buf[m++] = v_weighted_solid[j][1];
      //buf[m++] = v_weighted_solid[j][2];
      //buf[m++] = a_weighted_solid[j][0];
      //buf[m++] = a_weighted_solid[j][1];
      //buf[m++] = a_weighted_solid[j][2];
   
      for (int k = 0; k < 3; k++)
        for (int r = 0; r < 3; r++)
          buf[m++] = deviatoricTensor[j][k][r];

      buf[m++] = rhoI[j];
    }
  } else {
    if (domain->triclinic == 0) {
      dx = pbc[0] * domain->xprd;
      dy = pbc[1] * domain->yprd;
      dz = pbc[2] * domain->zprd;
    } else {
      dx = pbc[0] * domain->xprd + pbc[5] * domain->xy + pbc[4] * domain->xz;
      dy = pbc[1] * domain->yprd + pbc[3] * domain->yz;
      dz = pbc[2] * domain->zprd;
    }
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x[j][0] + dx;
      buf[m++] = x[j][1] + dy;
      buf[m++] = x[j][2] + dz;
      buf[m++] = v[j][0];
      buf[m++] = v[j][1];
      buf[m++] = v[j][2];
      buf[m++] = rho[j];
      buf[m++] = e[j];
      buf[m++] = vest[j][0];
      buf[m++] = vest[j][1];
      buf[m++] = vest[j][2];
      for (int k = 0; k < atom->num_sdpd_species; k++)  buf[m++] = C[j][k];

      if (atom->ssa_diffusion_flag == 1) {
        for (int k = 0; k < atom->num_ssa_species; k++)  buf[m++] = (double) Cd[j][k];
        buf[m++] = dfsp_D_matrix[j];
        buf[m++] = dfsp_D_diag[j];
        //buf[m++] = dfsp_Diffusion_coeff[j];
        buf[m++] = dfsp_a_i[j];      
      }

      if (atom->ssa_reaction_flag == 1) {
        for (int r = 0; r < atom->num_ssa_reactions; r++)  buf[m++] = ssa_rxn_propensity[j][r];
        for (int r = 0; r < atom->num_ssa_reactions; r++)
          for (int k = 0; k < atom->num_ssa_species; k++)
            buf[m++] = d_ssa_rxn_prop_d_c[j][r][k];
        for (int r = 0; r < atom->num_ssa_reactions; r++)
          for (int k = 0; k < atom->num_ssa_species; k++)
            buf[m++] = ssa_stoich_matrix[j][r][k];
      }

      //buf[m++] = phi[j];
      //buf[m++] = nw[j][0];
      //buf[m++] = nw[j][1];
      //buf[m++] = nw[j][2];
      //buf[m++] = v_weighted_solid[j][0];
      //buf[m++] = v_weighted_solid[j][1];
      //buf[m++] = v_weighted_solid[j][2];
      //buf[m++] = a_weighted_solid[j][0];
      //buf[m++] = a_weighted_solid[j][1];
      //buf[m++] = a_weighted_solid[j][2];

     for (int k = 0; k < 3; k++)
       for (int r = 0; r < 3; r++)
         buf[m++] = deviatoricTensor[j][k][r];

      buf[m++] = rhoI[j];
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void AtomVecSsaTsdpdAtomic::unpack_comm(int n, int first, double *buf) {

//  printf("in AtomVecSsaTsdpdAtomic::unpack_comm\n");

  int i, m, last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    x[i][0] = buf[m++];
    x[i][1] = buf[m++];
    x[i][2] = buf[m++];
    v[i][0] = buf[m++];
    v[i][1] = buf[m++];
    v[i][2] = buf[m++];
    rho[i] = buf[m++];
    e[i] = buf[m++];
    vest[i][0] = buf[m++];
    vest[i][1] = buf[m++];
    vest[i][2] = buf[m++];
    for (int k = 0; k < atom->num_sdpd_species; k++) C[i][k] = buf[m++];

    if (atom->ssa_diffusion_flag == 1) {
      for (int k = 0; k < atom->num_ssa_species; k++) Cd[i][k] = (int) buf[m++];
      dfsp_D_matrix[i] = buf[m++];
      dfsp_D_diag[i] = buf[m++];
      //dfsp_Diffusion_coeff[i] = buf[m++];
      dfsp_a_i[i] = buf[m++];
    }
 
    if (atom->ssa_reaction_flag == 1) {
      for (int r = 0; r < atom->num_ssa_reactions; r++) ssa_rxn_propensity[i][r] = buf[m++];
      for (int r = 0; r < atom->num_ssa_reactions; r++)
        for (int k = 0; k < atom->num_ssa_species; k++)
          d_ssa_rxn_prop_d_c[i][r][k] = buf[m++];
      for (int r = 0; r < atom->num_ssa_reactions; r++)
        for (int k = 0; k < atom->num_ssa_species; k++)
          ssa_stoich_matrix[i][r][k] = buf[m++];
    }
  

    //phi[i] = buf[m++];
    //nw[i][0] = buf[m++];
    //nw[i][1] = buf[m++];
    //nw[i][2] = buf[m++];
    //v_weighted_solid[i][0] = buf[m++];
    //v_weighted_solid[i][1] = buf[m++];
    //v_weighted_solid[i][2] = buf[m++];
    //a_weighted_solid[i][0] = buf[m++];
    //a_weighted_solid[i][1] = buf[m++];
    //a_weighted_solid[i][2] = buf[m++];

   for (int k = 0; k < 3; k++)
     for (int r = 0; r < 3; r++)
       deviatoricTensor[i][k][r] = buf[m++];

    rhoI[i] = buf[m++];
  }
}

/* ---------------------------------------------------------------------- */

void AtomVecSsaTsdpdAtomic::unpack_comm_vel(int n, int first, double *buf) {

//  printf("in AtomVecSsaTsdpdAtomic::unpack_comm_vel\n");

  int i, m, last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    x[i][0] = buf[m++];
    x[i][1] = buf[m++];
    x[i][2] = buf[m++];
    v[i][0] = buf[m++];
    v[i][1] = buf[m++];
    v[i][2] = buf[m++];
    rho[i] = buf[m++];
    e[i] = buf[m++];
    vest[i][0] = buf[m++];
    vest[i][1] = buf[m++];
    vest[i][2] = buf[m++];
    for (int k = 0; k < atom->num_sdpd_species; k++) C[i][k] = buf[m++];

    if (atom->ssa_diffusion_flag == 1) {
      for (int k = 0; k < atom->num_ssa_species; k++) Cd[i][k] = (int) buf[m++];
      dfsp_D_matrix[i] = buf[m++];
      dfsp_D_diag[i] = buf[m++];
      //dfsp_Diffusion_coeff[i] = buf[m++];
      dfsp_a_i[i] = buf[m++];
    }
 
    if (atom->ssa_reaction_flag == 1) {
      for (int r = 0; r < atom->num_ssa_reactions; r++) ssa_rxn_propensity[i][r] = buf[m++];
      for (int r = 0; r < atom->num_ssa_reactions; r++)
        for (int k = 0; k < atom->num_ssa_species; k++)
          d_ssa_rxn_prop_d_c[i][r][k] = buf[m++];
      for (int r = 0; r < atom->num_ssa_reactions; r++)
        for (int k = 0; k < atom->num_ssa_species; k++)
          ssa_stoich_matrix[i][r][k] = buf[m++];
    }

    //phi[i] = buf[m++];
    //nw[i][0] = buf[m++];
    //nw[i][1] = buf[m++];
    //nw[i][2] = buf[m++];
    //v_weighted_solid[i][0] = buf[m++];
    //v_weighted_solid[i][1] = buf[m++];
    //v_weighted_solid[i][2] = buf[m++];
    //a_weighted_solid[i][0] = buf[m++];
    //a_weighted_solid[i][1] = buf[m++];
    //a_weighted_solid[i][2] = buf[m++];

   for (int k = 0; k < 3; k++)
     for (int r = 0; r < 3; r++)
       deviatoricTensor[i][k][r] = buf[m++];

    rhoI[i] = buf[m++];
  }
}

/* ---------------------------------------------------------------------- */

int AtomVecSsaTsdpdAtomic::pack_reverse(int n, int first, double *buf) {

//  printf("in AtomVecSsaTsdpdAtomic::pack_reverse\n");

  int i, m, last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = f[i][0];
    buf[m++] = f[i][1];
    buf[m++] = f[i][2];
    buf[m++] = drho[i];
    buf[m++] = de[i];
    for (int k = 0; k < atom->num_sdpd_species; k++) buf[m++] = Q[i][k];

    if (atom->ssa_diffusion_flag == 1) {
      for (int k = 0; k < atom->num_ssa_species; k++) buf[m++] = (double) Qd[i][k];
    }

    for (int k = 0; k < 3; k++)
      for (int r = 0; r < 3; r++)
        buf[m++] = ddeviatoricTensor[i][k][r];

    for (int k = 0; k < 3; k++)
      for (int r = 0; r < 3; r++)
        buf[m++] = artificialStressTensor[i][k][r];

    //for (int k = 0; k < 3; k++)
    //  for (int r = 0; r < 3; r++)
    //    buf[m++] = kernelCorrectionTensor[i][k][r];

    buf[m++] = phi[i];
    buf[m++] = number_density[i];
    buf[m++] = nw[i][0];
    buf[m++] = nw[i][1];
    buf[m++] = nw[i][2];
    buf[m++] = v_weighted_solid[i][0];
    buf[m++] = v_weighted_solid[i][1];
    buf[m++] = v_weighted_solid[i][2];
    buf[m++] = a_weighted_solid[i][0];
    buf[m++] = a_weighted_solid[i][1];
    buf[m++] = a_weighted_solid[i][2];
    buf[m++] = ddx[i][0];
    buf[m++] = ddx[i][1];
    buf[m++] = ddx[i][2];
    buf[m++] = ddv[i][0];
    buf[m++] = ddv[i][1];
    buf[m++] = ddv[i][2];

    buf[m++] = Pold[i];
    buf[m++] = Pnew[i];
    buf[m++] = Aaux[i];
    buf[m++] = Baux[i];
    buf[m++] = APaux[i];
    buf[m++] = fP[i][0];
    buf[m++] = fP[i][1];
    buf[m++] = fP[i][2];
    buf[m++] = rhoAux1[i];
    buf[m++] = rhoAux2[i];
    buf[m++] = rhoAux3[i];
 
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void AtomVecSsaTsdpdAtomic::unpack_reverse(int n, int *list, double *buf) {

//  printf("in AtomVecSsaTsdpdAtomic::unpack_reverse\n");

  int i, j, m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    f[j][0] += buf[m++];
    f[j][1] += buf[m++];
    f[j][2] += buf[m++];
    drho[j] += buf[m++];
    de[j] += buf[m++];
    for (int k = 0; k < atom->num_sdpd_species; k++) Q[j][k] += buf[m++];
    
    if (atom->ssa_diffusion_flag == 1) {
      for (int k = 0; k < atom->num_ssa_species; k++) Qd[j][k] += (int) buf[m++];
    }

    for (int k = 0; k < 3; k++)
      for (int r = 0; r < 3; r++)
        ddeviatoricTensor[j][k][r] += buf[m++];

    for (int k = 0; k < 3; k++)
      for (int r = 0; r < 3; r++)
        artificialStressTensor[j][k][r] += buf[m++];

    //for (int k = 0; k < 3; k++)
    //  for (int r = 0; r < 3; r++)
    //    kernelCorrectionTensor[j][k][r] += buf[m++];

    phi[j]                 += buf[m++];
    number_density[j]      += buf[m++];
    nw[j][0]               += buf[m++];
    nw[j][1]               += buf[m++];
    nw[j][2]               += buf[m++];
    v_weighted_solid[j][0] += buf[m++];
    v_weighted_solid[j][1] += buf[m++];
    v_weighted_solid[j][2] += buf[m++];
    a_weighted_solid[j][0] += buf[m++];
    a_weighted_solid[j][1] += buf[m++];
    a_weighted_solid[j][2] += buf[m++];
    ddx[j][0]              += buf[m++];
    ddx[j][1]              += buf[m++];
    ddx[j][2]              += buf[m++];
    ddv[j][0]              += buf[m++];
    ddv[j][1]              += buf[m++];
    ddv[j][2]              += buf[m++];

    Pold[j]                += buf[m++];
    Pnew[j]                += buf[m++];
    Aaux[j]                += buf[m++];
    Baux[j]                += buf[m++];
    APaux[j]               += buf[m++];
    fP[j][0]               += buf[m++];
    fP[j][1]               += buf[m++];
    fP[j][2]               += buf[m++];
    rhoAux1[j]             += buf[m++];
    rhoAux2[j]             += buf[m++];
    rhoAux3[j]             += buf[m++];
  }
}

/* ---------------------------------------------------------------------- */

int AtomVecSsaTsdpdAtomic::pack_border(int n, int *list, double *buf, int pbc_flag,
                             int *pbc) {

//  printf("in AtomVecSsaTsdpdAtomic::pack_border\n");

  int i, j, m, r, k;
  double dx, dy, dz;

  m = 0;
  if (pbc_flag == 0) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x[j][0];
      buf[m++] = x[j][1];
      buf[m++] = x[j][2];
      buf[m++] = v[j][0]; //
      buf[m++] = v[j][1]; //
      buf[m++] = v[j][2]; //
      buf[m++] = ubuf(tag[j]).d;
      buf[m++] = ubuf(type[j]).d;
      buf[m++] = ubuf(mask[j]).d;
      //buf[m++] = ubuf(molecule[j]).d;
      buf[m++] = rho[j];
      buf[m++] = e[j];
      buf[m++] = cv[j];
      buf[m++] = vest[j][0];
      buf[m++] = vest[j][1];
      buf[m++] = vest[j][2];
      for (int k = 0; k < atom->num_sdpd_species; k++) buf[m++] = C[j][k];

      if (atom->ssa_diffusion_flag == 1) {
        for (int k = 0; k < atom->num_ssa_species; k++) buf[m++] = (double) Cd[j][k];
        buf[m++] = dfsp_D_matrix[j];
        buf[m++] = dfsp_D_diag[j];
        //buf[m++] = dfsp_Diffusion_coeff[j];
        buf[m++] = dfsp_a_i[j];
      }

      if (atom->ssa_reaction_flag == 1) {
        for (int r = 0; r < atom->num_ssa_reactions; r++)  buf[m++] = ssa_rxn_propensity[j][r];
        for (int r = 0; r < atom->num_ssa_reactions; r++)
          for (int k = 0; k < atom->num_ssa_species; k++)
            buf[m++] = d_ssa_rxn_prop_d_c[j][r][k];
        for (int r = 0; r < atom->num_ssa_reactions; r++)
          for (int k = 0; k < atom->num_ssa_species; k++)
            buf[m++] = ssa_stoich_matrix[j][r][k];
      }

      buf[m++] = ubuf(solid_tag[j]).d;
      buf[m++] = ubuf(fixed_tag[j]).d;
      //buf[m++] = phi[j];
      //buf[m++] = nw[j][0];
      //buf[m++] = nw[j][1];
      //buf[m++] = nw[j][2];
      //buf[m++] = v_weighted_solid[j][0];
      //buf[m++] = v_weighted_solid[j][1];
      //buf[m++] = v_weighted_solid[j][2];
      //buf[m++] = a_weighted_solid[j][0];
      //buf[m++] = a_weighted_solid[j][1];
      //buf[m++] = a_weighted_solid[j][2];

      for (int k = 0; k < 3; k++)
        for (int r = 0; r < 3; r++)
          buf[m++] = deviatoricTensor[j][k][r];

      buf[m++] = rhoI[j];
    }
  } else {
    if (domain->triclinic == 0) {
      dx = pbc[0] * domain->xprd;
      dy = pbc[1] * domain->yprd;
      dz = pbc[2] * domain->zprd;
    } else {
      dx = pbc[0];
      dy = pbc[1];
      dz = pbc[2];
    }
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x[j][0] + dx;
      buf[m++] = x[j][1] + dy;
      buf[m++] = x[j][2] + dz;
      buf[m++] = v[j][0]; //
      buf[m++] = v[j][1]; //
      buf[m++] = v[j][2]; //
      buf[m++] = ubuf(tag[j]).d;
      buf[m++] = ubuf(type[j]).d;
      buf[m++] = ubuf(mask[j]).d;
      buf[m++] = rho[j];
      buf[m++] = e[j];
      buf[m++] = cv[j];
      buf[m++] = vest[j][0];
      buf[m++] = vest[j][1];
      buf[m++] = vest[j][2];
      for (int k = 0; k < atom->num_sdpd_species; k++) buf[m++] = C[j][k];

      if (atom->ssa_diffusion_flag == 1) {
        for (int k = 0; k < atom->num_ssa_species; k++) buf[m++] = (double) Cd[j][k];
        buf[m++] = dfsp_D_matrix[j];
        buf[m++] = dfsp_D_diag[j];
        //buf[m++] = dfsp_Diffusion_coeff[j];
        buf[m++] = dfsp_a_i[j];
      }

      if (atom->ssa_reaction_flag == 1) {
        for (int r = 0; r < atom->num_ssa_reactions; r++)  buf[m++] = ssa_rxn_propensity[j][r];
        for (int r = 0; r < atom->num_ssa_reactions; r++)
          for (int k = 0; k < atom->num_ssa_species; k++)
            buf[m++] = d_ssa_rxn_prop_d_c[j][r][k];
        for (int r = 0; r < atom->num_ssa_reactions; r++)
          for (int k = 0; k < atom->num_ssa_species; k++)
            buf[m++] = ssa_stoich_matrix[j][r][k];
      }

      buf[m++] = ubuf(solid_tag[j]).d;
      buf[m++] = ubuf(fixed_tag[j]).d;
      //buf[m++] = phi[j];
      //buf[m++] = nw[j][0];
      //buf[m++] = nw[j][1];
      //buf[m++] = nw[j][2];
      //buf[m++] = v_weighted_solid[j][0];
      //buf[m++] = v_weighted_solid[j][1];
      //buf[m++] = v_weighted_solid[j][2];
      //buf[m++] = a_weighted_solid[j][0];
      //buf[m++] = a_weighted_solid[j][1];
      //buf[m++] = a_weighted_solid[j][2];

     for (int k = 0; k < 3; k++)
       for (int r = 0; r < 3; r++)
         buf[m++] = deviatoricTensor[j][k][r];

     buf[m++] = rhoI[j];
    }
  }

  if (atom->nextra_border)
    for (int iextra = 0; iextra < atom->nextra_border; iextra++)
      m += modify->fix[atom->extra_border[iextra]]->pack_border(n,list,&buf[m]);

  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecSsaTsdpdAtomic::pack_border_vel(int n, int *list, double *buf, int pbc_flag,
                                 int *pbc)
{
  int i,j,m, r,k;
  double dx,dy,dz,dvx,dvy,dvz;

  m = 0;
  if (pbc_flag == 0) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x[j][0];
      buf[m++] = x[j][1];
      buf[m++] = x[j][2];
      buf[m++] = ubuf(tag[j]).d;
      buf[m++] = ubuf(type[j]).d;
      buf[m++] = ubuf(mask[j]).d;
      buf[m++] = v[j][0];
      buf[m++] = v[j][1];
      buf[m++] = v[j][2];
      buf[m++] = rho[j];
      buf[m++] = e[j];
      buf[m++] = cv[j];
      buf[m++] = vest[j][0];
      buf[m++] = vest[j][1];
      buf[m++] = vest[j][2];
      for (int k = 0; k < atom->num_sdpd_species; k++) buf[m++] = C[j][k];

      if (atom->ssa_diffusion_flag == 1) {
        for (int k = 0; k < atom->num_ssa_species; k++) buf[m++] = (double) Cd[j][k];
        buf[m++] = dfsp_D_matrix[j];
        buf[m++] = dfsp_D_diag[j];
        //buf[m++] = dfsp_Diffusion_coeff[j];
        buf[m++] = dfsp_a_i[j];
      }

      if (atom->ssa_reaction_flag == 1) {
        for (int r = 0; r < atom->num_ssa_reactions; r++)  buf[m++] = ssa_rxn_propensity[j][r];
        for (int r = 0; r < atom->num_ssa_reactions; r++)
          for (int k = 0; k < atom->num_ssa_species; k++)
            buf[m++] = d_ssa_rxn_prop_d_c[j][r][k];
        for (int r = 0; r < atom->num_ssa_reactions; r++)
          for (int k = 0; k < atom->num_ssa_species; k++)
            buf[m++] = ssa_stoich_matrix[j][r][k];
      }

      buf[m++] = ubuf(solid_tag[j]).d;
      buf[m++] = ubuf(fixed_tag[j]).d;
      //buf[m++] = phi[j];
      //buf[m++] = nw[j][0];
      //buf[m++] = nw[j][1];
      //buf[m++] = nw[j][2];
      //buf[m++] = v_weighted_solid[j][0];
      //buf[m++] = v_weighted_solid[j][1];
      //buf[m++] = v_weighted_solid[j][2];
      //buf[m++] = a_weighted_solid[j][0];
      //buf[m++] = a_weighted_solid[j][1];
      //buf[m++] = a_weighted_solid[j][2];

      for (int k = 0; k < 3; k++)
        for (int r = 0; r < 3; r++)
          buf[m++] = deviatoricTensor[j][k][r];

    buf[m++] = rhoI[j];

    }
  } else {
    if (domain->triclinic == 0) {
      dx = pbc[0] * domain->xprd;
      dy = pbc[1] * domain->yprd;
      dz = pbc[2] * domain->zprd;
    } else {
      dx = pbc[0];
      dy = pbc[1];
      dz = pbc[2];
    }
    if (!deform_vremap) {
      for (i = 0; i < n; i++) {
        j = list[i];
        buf[m++] = x[j][0] + dx;
        buf[m++] = x[j][1] + dy;
        buf[m++] = x[j][2] + dz;
        buf[m++] = ubuf(tag[j]).d;
        buf[m++] = ubuf(type[j]).d;
        buf[m++] = ubuf(mask[j]).d;
        //buf[m++] = ubuf(molecule[j]).d;
        buf[m++] = v[j][0];
        buf[m++] = v[j][1];
        buf[m++] = v[j][2];
        buf[m++] = rho[j];
        buf[m++] = e[j];
        buf[m++] = cv[j];
        buf[m++] = vest[j][0];
        buf[m++] = vest[j][1];
        buf[m++] = vest[j][2];
        for (int k = 0; k < atom->num_sdpd_species; k++) buf[m++] = C[j][k];

  	if (atom->ssa_diffusion_flag == 1) {
          for (int k = 0; k < atom->num_ssa_species; k++) buf[m++] = (double) Cd[j][k];
          buf[m++] = dfsp_D_matrix[j];
          buf[m++] = dfsp_D_diag[j];
          //buf[m++] = dfsp_Diffusion_coeff[j];
          buf[m++] = dfsp_a_i[j];
        }

        if (atom->ssa_reaction_flag == 1) {
          for (int r = 0; r < atom->num_ssa_reactions; r++)  buf[m++] = ssa_rxn_propensity[j][r];
          for (int r = 0; r < atom->num_ssa_reactions; r++)
            for (int k = 0; k < atom->num_ssa_species; k++)
              buf[m++] = d_ssa_rxn_prop_d_c[j][r][k];
          for (int r = 0; r < atom->num_ssa_reactions; r++)
            for (int k = 0; k < atom->num_ssa_species; k++)
              buf[m++] = ssa_stoich_matrix[j][r][k];
        }

        buf[m++] = ubuf(solid_tag[j]).d;
        buf[m++] = ubuf(fixed_tag[j]).d;
        //buf[m++] = phi[j];
        //buf[m++] = nw[j][0];
        //buf[m++] = nw[j][1];
        //buf[m++] = nw[j][2];
        //buf[m++] = v_weighted_solid[j][0];
        //buf[m++] = v_weighted_solid[j][1];
        //buf[m++] = v_weighted_solid[j][2];
        //buf[m++] = a_weighted_solid[j][0];
        //buf[m++] = a_weighted_solid[j][1];
        //buf[m++] = a_weighted_solid[j][2];

        for (int k = 0; k < 3; k++)
          for (int r = 0; r < 3; r++)
            buf[m++] = deviatoricTensor[j][k][r];

      buf[m++] = rhoI[j];
      }
    } else {
      dvx = pbc[0] * h_rate[0] + pbc[5] * h_rate[5] + pbc[4] * h_rate[4];
      dvy = pbc[1] * h_rate[1] + pbc[3] * h_rate[3];
      dvz = pbc[2] * h_rate[2];
      for (i = 0; i < n; i++) {
        j = list[i];
        buf[m++] = x[j][0] + dx;
        buf[m++] = x[j][1] + dy;
        buf[m++] = x[j][2] + dz;
        buf[m++] = ubuf(tag[j]).d;
        buf[m++] = ubuf(type[j]).d;
        buf[m++] = ubuf(mask[j]).d;
        //buf[m++] = ubuf(molecule[j]).d;
        if (mask[i] & deform_groupbit) {
          buf[m++] = v[j][0] + dvx;
          buf[m++] = v[j][1] + dvy;
          buf[m++] = v[j][2] + dvz;
          buf[m++] = vest[j][0] + dvx;
          buf[m++] = vest[j][1] + dvy;
          buf[m++] = vest[j][2] + dvz;
        } else {
          buf[m++] = v[j][0];
          buf[m++] = v[j][1];
          buf[m++] = v[j][2];
          buf[m++] = vest[j][0];
          buf[m++] = vest[j][1];
          buf[m++] = vest[j][2];
        }
        buf[m++] = rho[j];
        buf[m++] = e[j];
        buf[m++] = cv[j];
        for (int k = 0; k < atom->num_sdpd_species; k++) buf[m++] = C[j][k];

	if (atom->ssa_diffusion_flag == 1) {
          for (int k = 0; k < atom->num_ssa_species; k++) buf[m++] = (double) Cd[j][k];
          buf[m++] = dfsp_D_matrix[j];
          buf[m++] = dfsp_D_diag[j];
          //buf[m++] = dfsp_Diffusion_coeff[j];
          buf[m++] = dfsp_a_i[j];
        }

        if (atom->ssa_reaction_flag == 1) {
          for (int r = 0; r < atom->num_ssa_reactions; r++)  buf[m++] = ssa_rxn_propensity[j][r];
          for (int r = 0; r < atom->num_ssa_reactions; r++)
            for (int k = 0; k < atom->num_ssa_species; k++)
              buf[m++] = d_ssa_rxn_prop_d_c[j][r][k];
          for (int r = 0; r < atom->num_ssa_reactions; r++)
            for (int k = 0; k < atom->num_ssa_species; k++)
              buf[m++] = ssa_stoich_matrix[j][r][k];
        }


        buf[m++] = ubuf(solid_tag[j]).d;
        buf[m++] = ubuf(fixed_tag[j]).d;
        //buf[m++] = phi[j];
        //buf[m++] = nw[j][0];
        //buf[m++] = nw[j][1];
        //buf[m++] = nw[j][2];
        //buf[m++] = v_weighted_solid[j][0];
        //buf[m++] = v_weighted_solid[j][1];
        //buf[m++] = v_weighted_solid[j][2];
        //buf[m++] = a_weighted_solid[j][0];
        //buf[m++] = a_weighted_solid[j][1];
        //buf[m++] = a_weighted_solid[j][2];

        for (int k = 0; k < 3; k++)
          for (int r = 0; r < 3; r++)
            buf[m++] = deviatoricTensor[j][k][r];

      buf[m++] = rhoI[j];
      }
    }
  }

  if (atom->nextra_border)
    for (int iextra = 0; iextra < atom->nextra_border; iextra++)
      m += modify->fix[atom->extra_border[iextra]]->pack_border(n,list,&buf[m]);

  return m;
}

/* ---------------------------------------------------------------------- */

void AtomVecSsaTsdpdAtomic::unpack_border(int n, int first, double *buf) {
  
//  printf("in AtomVecSsaTsdpdAtomic::unpack_border\n");

  int i, m, last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    if (i == nmax)
      grow(0);
    x[i][0] = buf[m++];
    x[i][1] = buf[m++];
    x[i][2] = buf[m++];
    v[i][0] = buf[m++]; //
    v[i][1] = buf[m++]; //
    v[i][2] = buf[m++]; //
    tag[i] = (tagint) ubuf(buf[m++]).i;
    type[i] = (int) ubuf(buf[m++]).i;
    mask[i] = (int) ubuf(buf[m++]).i;
    rho[i] = buf[m++];
    e[i] = buf[m++];
    cv[i] = buf[m++];
    vest[i][0] = buf[m++];
    vest[i][1] = buf[m++];
    vest[i][2] = buf[m++];
    for (int k = 0; k < atom->num_sdpd_species; k++) C[i][k] = buf[m++];

    if (atom->ssa_diffusion_flag == 1) {
      for (int k = 0; k < atom->num_ssa_species; k++) Cd[i][k] = (int) buf[m++];  
      dfsp_D_matrix[i] = buf[m++];
      dfsp_D_diag[i] = buf[m++];
      //dfsp_Diffusion_coeff[i] = buf[m++];
      dfsp_a_i[i] = buf[m++];
    }

    if (atom->ssa_reaction_flag == 1) {
      for (int r = 0; r < atom->num_ssa_reactions; r++) ssa_rxn_propensity[i][r] = buf[m++];
      for (int r = 0; r < atom->num_ssa_reactions; r++)
        for (int k = 0; k < atom->num_ssa_species; k++)
          d_ssa_rxn_prop_d_c[i][r][k] = buf[m++];
      for (int r = 0; r < atom->num_ssa_reactions; r++)
        for (int k = 0; k < atom->num_ssa_species; k++)
          ssa_stoich_matrix[i][r][k] = buf[m++];
    }

    solid_tag[i] = (int) ubuf(buf[m++]).i;
    fixed_tag[i] = (int) ubuf(buf[m++]).i;
    //phi[i] = buf[m++];
    //nw[i][0] = buf[m++];
    //nw[i][1] = buf[m++];
    //nw[i][2] = buf[m++];
    //v_weighted_solid[i][0] = buf[m++];
    //v_weighted_solid[i][1] = buf[m++];
    //v_weighted_solid[i][2] = buf[m++];
    //a_weighted_solid[i][0] = buf[m++];
    //a_weighted_solid[i][1] = buf[m++];
    //a_weighted_solid[i][2] = buf[m++];

    for (int k = 0; k < 3; k++)
      for (int r = 0; r < 3; r++)
        deviatoricTensor[i][k][r] = buf[m++];

    rhoI[i] = buf[m++];
  }

  if (atom->nextra_border)
    for (int iextra = 0; iextra < atom->nextra_border; iextra++)
      m += modify->fix[atom->extra_border[iextra]]->
        unpack_border(n,first,&buf[m]);
}

/* ---------------------------------------------------------------------- */

void AtomVecSsaTsdpdAtomic::unpack_border_vel(int n, int first, double *buf) {

  //printf("in AtomVecSsaTsdpdAtomic::unpack_border_vel\n");

  int i, m, last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    if (i == nmax)
      grow(0);
    x[i][0] = buf[m++];
    x[i][1] = buf[m++];
    x[i][2] = buf[m++];
    v[i][0] = buf[m++]; //
    v[i][1] = buf[m++]; //
    v[i][2] = buf[m++]; //
    tag[i] = (tagint) ubuf(buf[m++]).i;
    type[i] = (int) ubuf(buf[m++]).i;
    mask[i] = (int) ubuf(buf[m++]).i;
    //molecule[i] = (tagint) ubuf(buf[m++]).i;
    v[i][0] = buf[m++];
    v[i][1] = buf[m++];
    v[i][2] = buf[m++];
    rho[i] = buf[m++];
    e[i] = buf[m++];
    cv[i] = buf[m++];
    vest[i][0] = buf[m++];
    vest[i][1] = buf[m++];
    vest[i][2] = buf[m++];
 
    for (int k = 0; k < atom->num_sdpd_species; k++) C[i][k] = buf[m++];

    if (atom->ssa_diffusion_flag == 1) {
      for (int k = 0; k < atom->num_ssa_species; k++) Cd[i][k] = (int) buf[m++];  
      dfsp_D_matrix[i] = buf[m++];
      dfsp_D_diag[i] = buf[m++];
      //dfsp_Diffusion_coeff[i] = buf[m++];
      dfsp_a_i[i] = buf[m++];
    }

    if (atom->ssa_reaction_flag == 1) {
      for (int r = 0; r < atom->num_ssa_reactions; r++) ssa_rxn_propensity[i][r] = buf[m++];
      for (int r = 0; r < atom->num_ssa_reactions; r++)
        for (int k = 0; k < atom->num_ssa_species; k++)
          d_ssa_rxn_prop_d_c[i][r][k] = buf[m++];
      for (int r = 0; r < atom->num_ssa_reactions; r++)
        for (int k = 0; k < atom->num_ssa_species; k++)
          ssa_stoich_matrix[i][r][k] = buf[m++];
    }


    solid_tag[i] = (int) ubuf(buf[m++]).i;
    fixed_tag[i] = (int) ubuf(buf[m++]).i;
    //phi[i] = buf[m++];
    //nw[i][0] = buf[m++];
    //nw[i][1] = buf[m++];
    //nw[i][2] = buf[m++];
    //v_weighted_solid[i][0] = buf[m++];
    //v_weighted_solid[i][1] = buf[m++];
    //v_weighted_solid[i][2] = buf[m++];
    //a_weighted_solid[i][0] = buf[m++];
    //a_weighted_solid[i][1] = buf[m++];
    //a_weighted_solid[i][2] = buf[m++];

    for (int k = 0; k < 3; k++)
      for (int r = 0; r < 3; r++)
        deviatoricTensor[i][k][r] = buf[m++];

    rhoI[i] = buf[m++];
  }

  if (atom->nextra_border)
    for (int iextra = 0; iextra < atom->nextra_border; iextra++)
      m += modify->fix[atom->extra_border[iextra]]->
        unpack_border(n,first,&buf[m]);
}

/* ----------------------------------------------------------------------
   pack data for atom I for sending to another proc
   xyz must be 1st 3 values, so comm::exchange() can test on them
   ------------------------------------------------------------------------- */

int AtomVecSsaTsdpdAtomic::pack_exchange(int i, double *buf) {
  
//  printf("in AtomVecSsaTsdpdAtomic::pack_exchange\n");

  int r,k;
  
  int m = 1;
  buf[m++] = x[i][0];
  buf[m++] = x[i][1];
  buf[m++] = x[i][2];
  buf[m++] = v[i][0];
  buf[m++] = v[i][1];
  buf[m++] = v[i][2];
  buf[m++] = ubuf(tag[i]).d;
  buf[m++] = ubuf(type[i]).d;
  buf[m++] = ubuf(mask[i]).d;
  buf[m++] = ubuf(image[i]).d;
  
  /*
  buf[m++] = ubuf(molecule[i]).d;
  buf[m++] = ubuf(num_bond[i]).d;
  for (k = 0; k < num_bond[i]; k++) {
    buf[m++] = ubuf(bond_type[i][k]).d;
    buf[m++] = ubuf(bond_atom[i][k]).d;
  }
  buf[m++] = ubuf(num_angle[i]).d;
  for (k = 0; k < num_angle[i]; k++) {
    buf[m++] = ubuf(angle_type[i][k]).d;
    buf[m++] = ubuf(angle_atom1[i][k]).d;
    buf[m++] = ubuf(angle_atom2[i][k]).d;
    buf[m++] = ubuf(angle_atom3[i][k]).d;
  }
  buf[m++] = ubuf(nspecial[i][0]).d;
  buf[m++] = ubuf(nspecial[i][1]).d;
  buf[m++] = ubuf(nspecial[i][2]).d;
  for (k = 0; k < nspecial[i][2]; k++) buf[m++] = ubuf(special[i][k]).d;
  */
  
  buf[m++] = rho[i];
  buf[m++] = e[i];
  buf[m++] = cv[i];
  buf[m++] = vest[i][0];
  buf[m++] = vest[i][1];
  buf[m++] = vest[i][2];
  for (k = 0; k < atom->num_sdpd_species; k++) buf[m++] = C[i][k];
  
  if (atom->ssa_diffusion_flag == 1) {
    for (k = 0; k < atom->num_ssa_species; k++) buf[m++] = (double) Cd[i][k];
    buf[m++] = dfsp_D_matrix[i];
    buf[m++] = dfsp_D_diag[i];
    //buf[m++] = dfsp_Diffusion_coeff[i];
    buf[m++] = dfsp_a_i[i];
  }

  if (atom->ssa_reaction_flag == 1) {
    for (r = 0; r < atom->num_ssa_reactions; r++)  buf[m++] = ssa_rxn_propensity[i][r];
    for (r = 0; r < atom->num_ssa_reactions; r++)
      for (k = 0; k < atom->num_ssa_species; k++)
        buf[m++] = d_ssa_rxn_prop_d_c[i][r][k];
    for (r = 0; r < atom->num_ssa_reactions; r++)
      for (k = 0; k < atom->num_ssa_species; k++)
        buf[m++] = ssa_stoich_matrix[i][r][k];
  }

  buf[m++] = ubuf(solid_tag[i]).d;
  buf[m++] = ubuf(fixed_tag[i]).d;
  //buf[m++] = phi[i];
  //buf[m++] = nw[i][0];
  //buf[m++] = nw[i][1];
  //buf[m++] = nw[i][2];
  //buf[m++] = v_weighted_solid[i][0];
  //buf[m++] = v_weighted_solid[i][1];
  //buf[m++] = v_weighted_solid[i][2];
  //buf[m++] = a_weighted_solid[i][0];
  //buf[m++] = a_weighted_solid[i][1];
  //buf[m++] = a_weighted_solid[i][2];

  for (int k = 0; k < 3; k++)
    for (int r = 0; r < 3; r++)
      buf[m++] = deviatoricTensor[i][k][r];

  buf[m++] = rhoI[i];

  if (atom->nextra_grow)
    for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
      m += modify->fix[atom->extra_grow[iextra]]->pack_exchange(i, &buf[m]);

  buf[0] = m;
  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecSsaTsdpdAtomic::unpack_exchange(double *buf) {

  //printf("in AtomVecSsaTsdpdAtomic::unpack_exchange\n");
  
  int r, k;

  int nlocal = atom->nlocal;
  if (nlocal == nmax) grow(0);

  int m = 1;
  x[nlocal][0] = buf[m++];
  x[nlocal][1] = buf[m++];
  x[nlocal][2] = buf[m++];
  v[nlocal][0] = buf[m++];
  v[nlocal][1] = buf[m++];
  v[nlocal][2] = buf[m++];
  tag[nlocal] = (tagint) ubuf(buf[m++]).i;
  type[nlocal] = (int) ubuf(buf[m++]).i;
  mask[nlocal] = (int) ubuf(buf[m++]).i;
  image[nlocal] = (imageint) ubuf(buf[m++]).i;
  /*
  molecule[nlocal] = (tagint) ubuf(buf[m++]).i;
  num_bond[nlocal] = (int) ubuf(buf[m++]).i;
  for (k = 0; k < num_bond[nlocal]; k++) {
    bond_type[nlocal][k] = (int) ubuf(buf[m++]).i;
    bond_atom[nlocal][k] = (tagint) ubuf(buf[m++]).i;
  }
  num_angle[nlocal] = (int) ubuf(buf[m++]).i;
  for (k = 0; k < num_angle[nlocal]; k++) {
    angle_type[nlocal][k] = (int) ubuf(buf[m++]).i;
    angle_atom1[nlocal][k] = (tagint) ubuf(buf[m++]).i;
    angle_atom2[nlocal][k] = (tagint) ubuf(buf[m++]).i;
    angle_atom3[nlocal][k] = (tagint) ubuf(buf[m++]).i;
  }
  nspecial[nlocal][0] = (int) ubuf(buf[m++]).i;
  nspecial[nlocal][1] = (int) ubuf(buf[m++]).i;
  nspecial[nlocal][2] = (int) ubuf(buf[m++]).i;
  for (k = 0; k < nspecial[nlocal][2]; k++)
    special[nlocal][k] = (tagint) ubuf(buf[m++]).i;
  */
  rho[nlocal] = buf[m++];
  e[nlocal] = buf[m++];
  cv[nlocal] = buf[m++];
  vest[nlocal][0] = buf[m++];
  vest[nlocal][1] = buf[m++];
  vest[nlocal][2] = buf[m++];
  for (k = 0; k < atom->num_sdpd_species; k++) C[nlocal][k] = buf[m++];

  if (atom->ssa_diffusion_flag == 1) {
    for (k = 0; k < atom->num_ssa_species; k++) Cd[nlocal][k] = (int) buf[m++];
    dfsp_D_matrix[nlocal] = buf[m++];
    dfsp_D_diag[nlocal] = buf[m++];
    //dfsp_Diffusion_coeff[nlocal] = buf[m++];
    dfsp_a_i[nlocal] = buf[m++];
  }

  if (atom->ssa_reaction_flag == 1) {
    for (r = 0; r < atom->num_ssa_reactions; r++) ssa_rxn_propensity[nlocal][r] = buf[m++];
    for (r = 0; r < atom->num_ssa_reactions; r++)
      for (k = 0; k < atom->num_ssa_species; k++)
        d_ssa_rxn_prop_d_c[nlocal][r][k] = buf[m++];
    for (r = 0; r < atom->num_ssa_reactions; r++)
      for (k = 0; k < atom->num_ssa_species; k++)
        ssa_stoich_matrix[nlocal][r][k] = buf[m++];
  }

  solid_tag[nlocal] = (int) ubuf(buf[m++]).i;
  fixed_tag[nlocal] = (int) ubuf(buf[m++]).i;
  //phi[nlocal] = buf[m++];
  //nw[nlocal][0] = buf[m++];
  //nw[nlocal][1] = buf[m++];
  //nw[nlocal][2] = buf[m++];
  //v_weighted_solid[nlocal][0] = buf[m++];
  //v_weighted_solid[nlocal][1] = buf[m++];
  //v_weighted_solid[nlocal][2] = buf[m++];
  //a_weighted_solid[nlocal][0] = buf[m++];
  //a_weighted_solid[nlocal][1] = buf[m++];
  //a_weighted_solid[nlocal][2] = buf[m++];

  for (int k = 0; k < 3; k++)
    for (int r = 0; r < 3; r++)
      deviatoricTensor[nlocal][k][r] = buf[m++];

  rhoI[nlocal] = buf[m++];

  if (atom->nextra_grow)
    for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
      m += modify->fix[atom->extra_grow[iextra]]-> unpack_exchange(nlocal,
                                                                   &buf[m]);

  atom->nlocal++;
  return m;
}

/* ----------------------------------------------------------------------
   size of restart data for all atoms owned by this proc
   include extra data stored by fixes
   ------------------------------------------------------------------------- */

int AtomVecSsaTsdpdAtomic::size_restart() {
  int i;

  int nlocal = atom->nlocal;
  //int n = ( 17 +  atom->num_sdpd_species + atom->num_ssa_species + 2 * atom->num_ssa_reactions * atom->num_ssa_species + atom->num_ssa_reactions) * nlocal; // 11 + rho + e + cv + vest[3]
  int n = ( 17 +  atom->num_sdpd_species) * nlocal; // 11 + rho + e + cv + vest[3]
  
  //for (i = 0; i < nlocal; i++) n += 2*num_bond[i] + 4*num_angle[i];

  if (atom->nextra_restart)
    for (int iextra = 0; iextra < atom->nextra_restart; iextra++)
      for (i = 0; i < nlocal; i++)
        n += modify->fix[atom->extra_restart[iextra]]->size_restart(i);

  return n;
}

/* ----------------------------------------------------------------------
   pack atom I's data for restart file including extra quantities
   xyz must be 1st 3 values, so that read_restart can test on them
   molecular types may be negative, but write as positive
   ------------------------------------------------------------------------- */

int AtomVecSsaTsdpdAtomic::pack_restart(int i, double *buf) {
  int m = 1;
  int k,r;
  buf[m++] = x[i][0];
  buf[m++] = x[i][1];
  buf[m++] = x[i][2];
  buf[m++] = ubuf(tag[i]).d;
  buf[m++] = ubuf(type[i]).d;
  buf[m++] = ubuf(mask[i]).d;
  buf[m++] = ubuf(image[i]).d;
  /*
  buf[m++] = ubuf(molecule[i]).d;
  buf[m++] = ubuf(num_bond[i]).d;
  for (k = 0; k < num_bond[i]; k++) {
    buf[m++] = ubuf(MAX(bond_type[i][k],-bond_type[i][k])).d;
    buf[m++] = ubuf(bond_atom[i][k]).d;
  }
  buf[m++] = ubuf(num_angle[i]).d;
  for (k = 0; k < num_angle[i]; k++) {
    buf[m++] = ubuf(MAX(angle_type[i][k],-angle_type[i][k])).d;
    buf[m++] = ubuf(angle_atom1[i][k]).d;
    buf[m++] = ubuf(angle_atom2[i][k]).d;
    buf[m++] = ubuf(angle_atom3[i][k]).d;
  }
  */
  buf[m++] = v[i][0];
  buf[m++] = v[i][1];
  buf[m++] = v[i][2];
  buf[m++] = rho[i];
  buf[m++] = e[i];
  buf[m++] = cv[i];
  buf[m++] = vest[i][0];
  buf[m++] = vest[i][1];
  buf[m++] = vest[i][2];
  for (k = 0; k < atom->num_sdpd_species; k++) buf[m++] = C[i][k];

  if (atom->ssa_diffusion_flag == 1) {
    for (k = 0; k < atom->num_ssa_species; k++) buf[m++] = (double) Cd[i][k];
    buf[m++] = dfsp_D_matrix[i];
    buf[m++] = dfsp_D_diag[i];
    //buf[m++] = dfsp_Diffusion_coeff[i];
    buf[m++] = dfsp_a_i[i];
  }

  if (atom->ssa_reaction_flag == 1) {
    for (r = 0; r < atom->num_ssa_reactions; r++)  buf[m++] = ssa_rxn_propensity[i][r];
    for (r = 0; r < atom->num_ssa_reactions; r++)
      for (k = 0; k < atom->num_ssa_species; k++)
        buf[m++] = d_ssa_rxn_prop_d_c[i][r][k];
    for (r = 0; r < atom->num_ssa_reactions; r++)
      for (k = 0; k < atom->num_ssa_species; k++)
        buf[m++] = ssa_stoich_matrix[i][r][k];
  }

  buf[m++] = ubuf(solid_tag[i]).d;
  buf[m++] = ubuf(fixed_tag[i]).d;
  //buf[m++] = phi[i];
  //buf[m++] = nw[i][0];
  //buf[m++] = nw[i][1];
  //buf[m++] = nw[i][2];
  //buf[m++] = v_weighted_solid[i][0];
  //buf[m++] = v_weighted_solid[i][1];
  //buf[m++] = v_weighted_solid[i][2];
  //buf[m++] = a_weighted_solid[i][0];
  //buf[m++] = a_weighted_solid[i][1];
  //buf[m++] = a_weighted_solid[i][2];

  for (int k = 0; k < 3; k++)
    for (int r = 0; r < 3; r++)
      buf[m++] = deviatoricTensor[i][k][r];

  buf[m++] = rhoI[i];

  if (atom->nextra_restart)
    for (int iextra = 0; iextra < atom->nextra_restart; iextra++)
      m += modify->fix[atom->extra_restart[iextra]]->pack_restart(i, &buf[m]);

  buf[0] = m;
  return m;
}

/* ----------------------------------------------------------------------
   unpack data for one atom from restart file including extra quantities
   ------------------------------------------------------------------------- */

int AtomVecSsaTsdpdAtomic::unpack_restart(double *buf) {
  int nlocal = atom->nlocal;
  if (nlocal == nmax) {
    grow(0);
    if (atom->nextra_store)
      memory->grow(atom->extra, nmax, atom->nextra_store, "atom:extra");
  }

  int k,r;
  int m = 1;
  x[nlocal][0] = buf[m++];
  x[nlocal][1] = buf[m++];
  x[nlocal][2] = buf[m++];
  tag[nlocal] = (tagint) ubuf(buf[m++]).i;
  type[nlocal] = (int) ubuf(buf[m++]).i;
  mask[nlocal] = (int) ubuf(buf[m++]).i;
  image[nlocal] = (imageint) ubuf(buf[m++]).i;
  /*
  molecule[nlocal] = (tagint) ubuf(buf[m++]).i;
  num_bond[nlocal] = (int) ubuf(buf[m++]).i;
  for (k = 0; k < num_bond[nlocal]; k++) {
    bond_type[nlocal][k] = (int) ubuf(buf[m++]).i;
    bond_atom[nlocal][k] = (tagint) ubuf(buf[m++]).i;
  }
  num_angle[nlocal] = (int) ubuf(buf[m++]).i;
  for (k = 0; k < num_angle[nlocal]; k++) {
    angle_type[nlocal][k] = (int) ubuf(buf[m++]).i;
    angle_atom1[nlocal][k] = (tagint) ubuf(buf[m++]).i;
    angle_atom2[nlocal][k] = (tagint) ubuf(buf[m++]).i;
    angle_atom3[nlocal][k] = (tagint) ubuf(buf[m++]).i;
  }
  nspecial[nlocal][0] = nspecial[nlocal][1] = nspecial[nlocal][2] = 0;
  */
  
  v[nlocal][0] = buf[m++];
  v[nlocal][1] = buf[m++];
  v[nlocal][2] = buf[m++];
  rho[nlocal] = buf[m++];
  e[nlocal] = buf[m++];
  cv[nlocal] = buf[m++];
  vest[nlocal][0] = buf[m++];
  vest[nlocal][1] = buf[m++];
  vest[nlocal][2] = buf[m++];
  for(k = 0; k < atom->num_sdpd_species; k++) C[nlocal][k] = buf[m++];

  if (atom->ssa_diffusion_flag == 1) {
    for(k = 0; k < atom->num_ssa_species; k++) Cd[nlocal][k] = buf[m++];  
    dfsp_D_matrix[nlocal] = buf[m++];
    dfsp_D_diag[nlocal] = buf[m++];
    //dfsp_Diffusion_coeff[nlocal] = buf[m++];
    dfsp_a_i[nlocal] = buf[m++];
  }

  if (atom->ssa_reaction_flag == 1) {
    for (r = 0; r < atom->num_ssa_reactions; r++) ssa_rxn_propensity[nlocal][r] = buf[m++];
    for (r = 0; r < atom->num_ssa_reactions; r++)
      for (k = 0; k < atom->num_ssa_species; k++)
        d_ssa_rxn_prop_d_c[nlocal][r][k] = buf[m++];
    for (r = 0; r < atom->num_ssa_reactions; r++)
      for (k = 0; k < atom->num_ssa_species; k++)
        ssa_stoich_matrix[nlocal][r][k] = buf[m++];
  }
 
  solid_tag[nlocal] = (int) ubuf(buf[m++]).i;
  fixed_tag[nlocal] = (int) ubuf(buf[m++]).i;
  //phi[nlocal] = buf[m++];
  //nw[nlocal][0] = buf[m++];
  //nw[nlocal][1] = buf[m++];
  //nw[nlocal][2] = buf[m++];
  //v_weighted_solid[nlocal][0] = buf[m++];
  //v_weighted_solid[nlocal][1] = buf[m++];
  //v_weighted_solid[nlocal][2] = buf[m++];
  //a_weighted_solid[nlocal][0] = buf[m++];
  //a_weighted_solid[nlocal][1] = buf[m++];
  //a_weighted_solid[nlocal][2] = buf[m++];

  for (int k = 0; k < 3; k++)
    for (int r = 0; r < 3; r++)
      deviatoricTensor[nlocal][k][r] = buf[m++];

  rhoI[nlocal] = buf[m++];

  double **extra = atom->extra;
  if (atom->nextra_store) {
    int size = static_cast<int> (buf[0]) - m;
    for (int i = 0; i < size; i++)
      extra[nlocal][i] = buf[m++];
  }

  atom->nlocal++;
  return m;
}

/* ----------------------------------------------------------------------
   create one atom of itype at coord
   set other values to defaults
   ------------------------------------------------------------------------- */

void AtomVecSsaTsdpdAtomic::create_atom(int itype, double *coord) {
  int nlocal = atom->nlocal;
  if (nlocal == nmax)
    grow(0);

  tag[nlocal] = 0;
  type[nlocal] = itype;
  x[nlocal][0] = coord[0];
  x[nlocal][1] = coord[1];
  x[nlocal][2] = coord[2];
  mask[nlocal] = 1;
  image[nlocal] = ((imageint) IMGMAX << IMG2BITS) |
    ((imageint) IMGMAX << IMGBITS) | IMGMAX;
  v[nlocal][0] = 0.0;
  v[nlocal][1] = 0.0;
  v[nlocal][2] = 0.0;
  /*
  molecule[nlocal] = 0;
  num_bond[nlocal] = 0;
  num_angle[nlocal] = 0;
  nspecial[nlocal][0] = nspecial[nlocal][1] = nspecial[nlocal][2] = 0;
  */
  rho[nlocal] = 0.0;
  e[nlocal] = 0.0;
  cv[nlocal] = 1.0;
  vest[nlocal][0] = 0.0;
  vest[nlocal][1] = 0.0;
  vest[nlocal][2] = 0.0;
  de[nlocal] = 0.0;
  drho[nlocal] = 0.0;
  for (int k = 0; k < atom->num_sdpd_species; k++) C[nlocal][k] = 0.0;
  for (int k = 0; k < atom->num_sdpd_species; k++) Q[nlocal][k] = 0.0;

  if (atom->ssa_diffusion_flag == 1) {
    for (int k = 0; k < atom->num_ssa_species; k++) Cd[nlocal][k] = 0;
  }

  if (atom->ssa_reaction_flag == 1) {
    for (int r = 0; r < atom->num_ssa_reactions; r++) ssa_rxn_propensity[nlocal][r] = 0.0;
    for (int r = 0; r < atom->num_ssa_reactions; r++)
      for (int k = 0; k < atom->num_ssa_species; k++)
        d_ssa_rxn_prop_d_c[nlocal][r][k] = 0.0;
    for (int r = 0; r < atom->num_ssa_reactions; r++)
      for (int k = 0; k < atom->num_ssa_species; k++)
        ssa_stoich_matrix[nlocal][r][k] = 0.0;
  }

  solid_tag[nlocal] = 0;
  fixed_tag[nlocal] = 0;
  phi[nlocal] = 0.0;
  number_density[nlocal] = 0.0;
  nw[nlocal][0] = 0.0;
  nw[nlocal][1] = 0.0;
  nw[nlocal][2] = 0.0;
  v_weighted_solid[nlocal][0] = 0.0;
  v_weighted_solid[nlocal][1] = 0.0;
  v_weighted_solid[nlocal][2] = 0.0;
  a_weighted_solid[nlocal][0] = 0.0;
  a_weighted_solid[nlocal][1] = 0.0;
  a_weighted_solid[nlocal][2] = 0.0;

  for (int k = 0; k < 3; k++) {
    for (int r = 0; r < 3; r++) {
      deviatoricTensor[nlocal][k][r] = 0.0;
      ddeviatoricTensor[nlocal][k][r] = 0.0; 
      artificialStressTensor[nlocal][k][r] = 0.0; 
      //kernelCorrectionTensor[nlocal][k][r] = 0.0; 
    }
  }
  
  ddx[nlocal][0] = 0.0;
  ddx[nlocal][1] = 0.0;
  ddx[nlocal][2] = 0.0;
  ddv[nlocal][0] = 0.0;
  ddv[nlocal][1] = 0.0;
  ddv[nlocal][2] = 0.0;
  
  Pold[nlocal] = 0.0;
  Pnew[nlocal] = 0.0;
  Aaux[nlocal] = 0.0;
  Baux[nlocal] = 0.0;
  APaux[nlocal] = 0.0;
  fP[nlocal][0] = 0.0;
  fP[nlocal][1] = 0.0;
  fP[nlocal][2] = 0.0;
  rhoI[nlocal] = 0.0;
  rhoAux1[nlocal] = 0.0;
  rhoAux2[nlocal] = 0.0;
  rhoAux3[nlocal] = 0.0;

  atom->nlocal++;
}

/* ----------------------------------------------------------------------
   unpack one line from Atoms section of data file
   initialize other atom quantities
   ------------------------------------------------------------------------- */

void AtomVecSsaTsdpdAtomic::data_atom(double *coord, imageint imagetmp, char **values) {

  int nlocal = atom->nlocal;
  if (nlocal == nmax) grow(0);

  tag[nlocal] = ATOTAGINT(values[0]);
  solid_tag[nlocal] = ATOTAGINT(values[1]); 
  fixed_tag[nlocal] = 0; 
  type[nlocal] = atoi(values[2]);

  if (type[nlocal] <= 0 || type[nlocal] > atom->ntypes)
    error->one(FLERR,"Invalid atom type in Atoms section of data file");

  rho[nlocal] = atof(values[3]);

  x[nlocal][0] = coord[0];
  x[nlocal][1] = coord[1];
  x[nlocal][2] = coord[2];

//  int m = 5;

//  for (int k = 0; k < atom->num_sdpd_species; k++) C[nlocal][k] = atof(values[m+k]);
//  m += atom->num_sdpd_species;

//  for (int k = 0; k < atom->num_ssa_species; k++) Cd[nlocal][k] = atof(values[m+k]);
//  m += atom->num_ssa_species;

//  for (int r = 0; r < atom->num_ssa_reactions; r++) ssa_rxn_propensity[nlocal][r] = atof(values[m+r]);
//  m += atom->num_ssa_reactions;

//  for (int r = 0; r < atom->num_ssa_reactions; r++)
//      for (int k = 0; k < atom->num_ssa_species; k++)
//        d_ssa_rxn_prop_d_c[nlocal][r][k] = atof(values[m+r*atom->num_ssa_reactions+k]);
//  m += atom->num_ssa_reactions*atom->num_ssa_species;

  image[nlocal] = imagetmp;

  mask[nlocal] = 1;
  v[nlocal][0] = 0.0;
  v[nlocal][1] = 0.0;
  v[nlocal][2] = 0.0;

  vest[nlocal][0] = 0.0;
  vest[nlocal][1] = 0.0;
  vest[nlocal][2] = 0.0;

  de[nlocal] = 0.0;
  drho[nlocal] = 0.0;

  for (int k = 0; k < atom->num_sdpd_species; k++) C[nlocal][k] = 0.0;
  for (int k = 0; k < atom->num_sdpd_species; k++) Q[nlocal][k] = 0.0;

  //for (int k = 0; k < atom->num_ssa_species; k++) Cd[nlocal][k] = 0;
  /*
  for (int r = 0; r < atom->num_ssa_reactions; r++) ssa_rxn_propensity[nlocal][r] = 0.0;

  for (int r = 0; r < atom->num_ssa_reactions; r++)
    for (int k = 0; k < atom->num_ssa_species; k++)
      d_ssa_rxn_prop_d_c[nlocal][r][k] = 0.0;
  */

 

  for (int k = 0; k < 3; k++) {
    for (int r = 0; r < 3; r++) {
      deviatoricTensor[nlocal][k][r] = 0.0;
      ddeviatoricTensor[nlocal][k][r] = 0.0;
      artificialStressTensor[nlocal][k][r] = 0.0;
      //kernelCorrectionTensor[nlocal][k][r] = 0.0;
    }
  }

  ddx[nlocal][0] = 0.0;
  ddx[nlocal][1] = 0.0;
  ddx[nlocal][2] = 0.0;
  ddv[nlocal][0] = 0.0;
  ddv[nlocal][1] = 0.0;
  ddv[nlocal][2] = 0.0;

  Pold[nlocal] = 0.0;
  Pnew[nlocal] = 0.0;
  Aaux[nlocal] = 0.0;
  Baux[nlocal] = 0.0;
  APaux[nlocal] = 0.0;
  fP[nlocal][0] = 0.0;
  fP[nlocal][1] = 0.0;
  fP[nlocal][2] = 0.0;
  rhoI[nlocal] = 0.0;
  rhoAux1[nlocal] = 0.0;
  rhoAux2[nlocal] = 0.0;
  rhoAux3[nlocal] = 0.0;
  
  atom->nlocal++;
}

/* ----------------------------------------------------------------------
   pack atom info for data file including 3 image flags
------------------------------------------------------------------------- */

void AtomVecSsaTsdpdAtomic::pack_data(double **buf)
{
  int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++) {
    buf[i][0] = ubuf(tag[i]).d;
    //buf[i][1] = ubuf(molecule[i]).d;
    buf[i][2] = ubuf(type[i]).d;
    buf[i][3] = rho[i];
    buf[i][4] = e[i];
    buf[i][5] = cv[i];
    buf[i][3] = x[i][0];
    buf[i][4] = x[i][1];
    buf[i][5] = x[i][2];
    buf[i][6] = ubuf((image[i] & IMGMASK) - IMGMAX).d;
    buf[i][7] = ubuf((image[i] >> IMGBITS & IMGMASK) - IMGMAX).d;
    buf[i][8] = ubuf((image[i] >> IMG2BITS) - IMGMAX).d;
    buf[i][9] = ubuf(solid_tag[i]).d;
  }
}


/* ----------------------------------------------------------------------
   write atom info to data file including 3 image flags
------------------------------------------------------------------------- */

void AtomVecSsaTsdpdAtomic::write_data(FILE *fp, int n, double **buf)
{
  for (int i = 0; i < n; i++)
    fprintf(fp,TAGINT_FORMAT
            " %d %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e "
            "%d %d %d\n",
            (tagint) ubuf(buf[i][0]).i,(tagint) ubuf(buf[i][1]).i,
            (int) ubuf(buf[i][1]).i,
            buf[i][2],buf[i][3],buf[i][4],
            buf[i][5],buf[i][6],buf[i][7],
            (int) ubuf(buf[i][8]).i,(int) ubuf(buf[i][9]).i,
            (int) ubuf(buf[i][10]).i);
}


/* ----------------------------------------------------------------------
   assign an index to named atom property and return index
   return -1 if name is unknown to this atom style
------------------------------------------------------------------------- */

int AtomVecSsaTsdpdAtomic::property_atom(char *name)
{
  if (strcmp(name,"rho") == 0) return 0;
  if (strcmp(name,"drho") == 0) return 1;
  if (strcmp(name,"e") == 0) return 2;
  if (strcmp(name,"de") == 0) return 3;
  if (strcmp(name,"cv") == 0) return 4;
  if (strcmp(name,"phi") == 0) return 5;
  if (strcmp(name,"solid_tag") == 0) return 6;
  if (strcmp(name,"Pnew") == 0) return 7;
  if (strcmp(name,"deviatoricTensor") == 0) return 8;
  return -1;
}

/* ----------------------------------------------------------------------
   pack per-atom data into buf for ComputePropertyAtom
   index maps to data specific to this atom style
------------------------------------------------------------------------- */

void AtomVecSsaTsdpdAtomic::pack_property_atom(int index, double *buf,
                                     int nvalues, int groupbit)
{
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int n = 0;

  if (index == 0) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) buf[n] = rho[i];
      else buf[n] = 0.0;
      n += nvalues;
    }
  } else if (index == 1) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) buf[n] = drho[i];
      else buf[n] = 0.0;
      n += nvalues;
    }
  } else if (index == 2) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) buf[n] = e[i];
      else buf[n] = 0.0;
      n += nvalues;
    }
  } else if (index == 3) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) buf[n] = de[i];
      else buf[n] = 0.0;
      n += nvalues;
    }
  } else if (index == 4) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) buf[n] = cv[i];
      else buf[n] = 0.0;
      n += nvalues;
    }
  } else if (index == 5) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) buf[n] = phi[i];
      else buf[n] = 0.0;
      n += nvalues;
    }
  } else if (index == 6) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) buf[n] = solid_tag[i];
      else buf[n] = 0.0;
      n += nvalues;
    }
  } else if (index == 7) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) buf[n] = Pnew[i];
      else buf[n] = 0.0;
      n += nvalues;
    }
  } else if (index == 8) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        for (int k = 0; k < 3; k++)
          for (int r = 0; r < 3; r++)
            buf[n] = deviatoricTensor[i][k][r];
      }
      else buf[n] = 0.0;
      n += nvalues;
    }
  }
}

/* ----------------------------------------------------------------------
   return # of bytes of allocated memory
   ------------------------------------------------------------------------- */

bigint AtomVecSsaTsdpdAtomic::memory_usage() {
  bigint bytes = 0;

//  printf("in AtomVecSsaTsdpdAtomic::memory_usage\n");

  if (atom->memcheck("tag"))
    bytes += memory->usage(tag, nmax);
  if (atom->memcheck("type"))
    bytes += memory->usage(type, nmax);
  if (atom->memcheck("mask"))
    bytes += memory->usage(mask, nmax);
  if (atom->memcheck("image"))
    bytes += memory->usage(image, nmax);
  if (atom->memcheck("x"))
    bytes += memory->usage(x, nmax, 3);
  if (atom->memcheck("v"))
    bytes += memory->usage(v, nmax, 3);
  if (atom->memcheck("f"))
    bytes += memory->usage(f, nmax*comm->nthreads, 3);
  /*
  if (atom->memcheck("molecule")) 
    bytes += memory->usage(molecule,nmax);
  if (atom->memcheck("nspecial")) 
    bytes += memory->usage(nspecial,nmax,3);
  if (atom->memcheck("special"))  
    bytes += memory->usage(special,nmax,atom->maxspecial);
  if (atom->memcheck("num_bond")) 
    bytes += memory->usage(num_bond,nmax);
  if (atom->memcheck("bond_type")) 
    bytes += memory->usage(bond_type,nmax,atom->bond_per_atom);
  if (atom->memcheck("bond_atom")) 
    bytes += memory->usage(bond_atom,nmax,atom->bond_per_atom);
  if (atom->memcheck("num_angle")) 
    bytes += memory->usage(num_angle,nmax);
  if (atom->memcheck("angle_type")) 
    bytes += memory->usage(angle_type,nmax,atom->angle_per_atom);
  if (atom->memcheck("angle_atom1")) 
    bytes += memory->usage(angle_atom1,nmax,atom->angle_per_atom);
  if (atom->memcheck("angle_atom2")) 
    bytes += memory->usage(angle_atom2,nmax,atom->angle_per_atom);
  if (atom->memcheck("angle_atom3")) 
    bytes += memory->usage(angle_atom3,nmax,atom->angle_per_atom);
  */ 
  if (atom->memcheck("rho")) 
    bytes += memory->usage(rho, nmax);
  if (atom->memcheck("drho"))
    bytes += memory->usage(drho, nmax*comm->nthreads);
  if (atom->memcheck("e"))
    bytes += memory->usage(e, nmax);
  if (atom->memcheck("de"))
    bytes += memory->usage(de, nmax*comm->nthreads);
  if (atom->memcheck("cv"))
    bytes += memory->usage(cv, nmax);
  if (atom->memcheck("vest"))
    bytes += memory->usage(vest, nmax, 3);
  if (atom->memcheck("C")) 
    bytes += memory->usage(C,nmax,atom->num_sdpd_species);
  if (atom->memcheck("Q")) 
    bytes += memory->usage(Q,nmax*comm->nthreads,atom->num_sdpd_species);


  if (atom->ssa_diffusion_flag == 1) {
    if (atom->memcheck("Cd")) 
      bytes += memory->usage(Cd,nmax,atom->num_ssa_species);
    if (atom->memcheck("Qd")) 
      bytes += memory->usage(Qd,nmax*comm->nthreads,atom->num_ssa_species);
    if (atom->memcheck("dfsp_D_matrix"))
      bytes += memory->usage(dfsp_D_matrix, nmax);
    if (atom->memcheck("dfsp_D_diag"))
      bytes += memory->usage(dfsp_D_diag, nmax);
    //if (atom->memcheck("dfsp_Diffusion_coeff"))
    //  bytes += memory->usage(dfsp_Diffusion_coeff, nmax*nmax*atom->num_ssa_species);
    if (atom->memcheck("dfsp_a_i"))
      bytes += memory->usage(dfsp_a_i, nmax);
  }

  if (atom->ssa_reaction_flag == 1) {
    if (atom->memcheck("ssa_rxn_propensity")) 
      bytes += memory->usage(ssa_rxn_propensity,nmax,atom->num_ssa_reactions);
    if (atom->memcheck("d_ssa_rxn_prop_d_c")) 
      bytes += memory->usage(d_ssa_rxn_prop_d_c,nmax,atom->num_ssa_reactions,atom->num_ssa_species);
    if (atom->memcheck("ssa_stoich_matrix")) 
      bytes += memory->usage(ssa_stoich_matrix,nmax,atom->num_ssa_reactions,atom->num_ssa_species);
  }

  if (atom->memcheck("solid_tag"))
    bytes += memory->usage(solid_tag, nmax);
  if (atom->memcheck("fixed_tag"))
    bytes += memory->usage(fixed_tag, nmax); 
  if (atom->memcheck("phi")) 
    bytes += memory->usage(phi, nmax*comm->nthreads);
  if (atom->memcheck("number_density")) 
    bytes += memory->usage(number_density, nmax*comm->nthreads);
  if (atom->memcheck("nw")) 
    bytes += memory->usage(nw, nmax*comm->nthreads, 3);
  if (atom->memcheck("v_weighted_solid")) 
    bytes += memory->usage(v_weighted_solid, nmax*comm->nthreads, 3);
  if (atom->memcheck("a_weighted_solid")) 
    bytes += memory->usage(a_weighted_solid, nmax*comm->nthreads, 3);
  if (atom->memcheck("deviatoricTensor")) 
    bytes += memory->usage(deviatoricTensor, nmax, 3, 3);
  if (atom->memcheck("ddeviatoricTensor")) 
    bytes += memory->usage(ddeviatoricTensor, nmax*comm->nthreads, 3, 3);
  if (atom->memcheck("artificialStressTensor")) 
    bytes += memory->usage(artificialStressTensor, nmax*comm->nthreads, 3, 3);
  //if (atom->memcheck("kernelCorrectionTensor")) 
  //  bytes += memory->usage(kernelCorrectionTensor, nmax*comm->nthreads, 3, 3);
 if (atom->memcheck("ddv")) 
    bytes += memory->usage(ddv, nmax*comm->nthreads, 3);
 if (atom->memcheck("ddx")) 
    bytes += memory->usage(ddx, nmax*comm->nthreads, 3);

 if (atom->memcheck("Pold")) 
    bytes += memory->usage(Pold, nmax*comm->nthreads);
 if (atom->memcheck("Pnew")) 
    bytes += memory->usage(Pnew, nmax*comm->nthreads);
 if (atom->memcheck("Aaux")) 
    bytes += memory->usage(Aaux, nmax*comm->nthreads);
 if (atom->memcheck("Baux")) 
    bytes += memory->usage(Baux, nmax*comm->nthreads);
 if (atom->memcheck("APaux")) 
    bytes += memory->usage(APaux, nmax*comm->nthreads);
 if (atom->memcheck("fP")) 
    bytes += memory->usage(fP, nmax*comm->nthreads, 3);
 if (atom->memcheck("rhoI")) 
    bytes += memory->usage(rhoI, nmax);
 if (atom->memcheck("rhoAux1"))
    bytes += memory->usage(rhoAux1, nmax*comm->nthreads);
 if (atom->memcheck("rhoAux2"))
    bytes += memory->usage(rhoAux2, nmax*comm->nthreads);
 if (atom->memcheck("rhoAux3"))
    bytes += memory->usage(rhoAux3, nmax*comm->nthreads);

  return bytes;
}
