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
#include "fix_ssa_tsdpd_bvf_fsi.h"
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
#include "math_const.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

FixSsaTsdpdBvfFsi::FixSsaTsdpdBvfFsi(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg) {

    if ((atom->e_flag != 1) || (atom->rho_flag != 1))
        error->all(FLERR,"fix ssa_tsdpd/bvf command requires atom_style with both energy and density");

    if (narg != 3)
        error->all(FLERR,"Illegal number of arguments for fix ssa_tsdpd/bvf command");

    time_integrate = 1;

    // seed is immune to underflow/overflow because it is unsigned
    seed = comm->nprocs + comm->me + atom->nlocal;
    //if (narg == 3) seed += force->inumeric (FLERR, arg[2]);
    random = new RanMars (lmp, seed);

}

/* ---------------------------------------------------------------------- */

int FixSsaTsdpdBvfFsi::setmask() {
    int mask = 0;
    mask |= INITIAL_INTEGRATE;
    mask |= PRE_FORCE;
    mask |= FINAL_INTEGRATE;
    return mask;
}

/* ---------------------------------------------------------------------- */

void FixSsaTsdpdBvfFsi::init() {
    dtv = update->dt;
    dtf = 0.5 * update->dt * force->ftm2v;
}

/* ---------------------------------------------------------------------- */

void FixSsaTsdpdBvfFsi::setup_pre_force(int vflag)
{
    // set vest equal to v
    double **v = atom->v;
    double **vest = atom->vest;
    double *rhoI = atom->rhoI;
    double *rho = atom->rho;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    if (igroup == atom->firstgroup)
        nlocal = atom->nfirst;

    for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
            vest[i][0] = v[i][0];
            vest[i][1] = v[i][1];
            vest[i][2] = v[i][2];
	    rhoI[i] = rho[i];
        }
    }
}

/* ----------------------------------------------------------------------
 allow for both per-type and per-atom mass
 ------------------------------------------------------------------------- */

void FixSsaTsdpdBvfFsi::initial_integrate(int vflag) {

    // update v and x and rho and e of atoms in group
    double **x = atom->x;
    double **v = atom->v;
    double **f = atom->f;
    double **vest = atom->vest;
    double *rho = atom->rho;
    double *rhoI = atom->rhoI;
    double *drho = atom->drho;
    double *e = atom->e;
    double *de = atom->de;
    double *mass = atom->mass;
    double *rmass = atom->rmass;
    int rmass_flag = atom->rmass_flag;
    double dtCm;
    double **C = atom->C;
    double **Q = atom->Q;
    int *type = atom->type;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    int i,j;
    double dtfm;
    double enx,eny,enz,norm_nw,v_dot_en;
    double Ux_solid_nearby, Uy_solid_nearby, Uz_solid_nearby;
    double ax_solid_nearby, ay_solid_nearby, az_solid_nearby;
    int *solid_tag = atom->solid_tag;
    double *phi = atom->phi;
    double *number_density = atom->number_density;
    double **nw = atom->nw;
    double **v_weighted_solid = atom->v_weighted_solid;
    double **a_weighted_solid = atom->a_weighted_solid;
    double ***ddeviatoricTensor = atom->ddeviatoricTensor;
    double ***deviatoricTensor = atom->deviatoricTensor;
    double **ddv = atom->ddv;
    double **ddx = atom->ddx;
    double *rhoAux1 = atom->rhoAux1;
    double *rhoAux2 = atom->rhoAux2;
 
    double tnow = (double) update->ntimestep;
    double tmax = (double) update->nsteps;
    double tdamp = 1;
    double damp;
    if (tnow <= tdamp) damp = tnow/tdamp; //*( -0.5 * sin( (-0.5 + tnow/tdamp) * MY_PI ) + 0.5  );
    else damp = 1.0;
    
    double dampSolid;
    double tdampSolid = 1; //1e6;
    if (tnow <= tdampSolid) dampSolid = 0.0; //tnow/tdampSolid;
    else dampSolid = 1.0;

 
    if (igroup == atom->firstgroup)
        nlocal = atom->nfirst;

    for (i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
            if (rmass_flag) {
                dtfm = dtf / rmass[i];
            } else {
                dtfm = dtf / mass[type[i]];
            }

	    if (atom -> fixed_tag[i] == 0) { //then solid particles are free to move   (velocity = 0, position = x0, acceleration = 0)
                
	        if (solid_tag[i] == 0) { //if fluid particle ...        
	  	  // Compute intermediate (momentum) velocity     
                  vest[i][0] = v[i][0] + dtfm * f[i][0] * damp + 0.001 * ddx[i][0]/number_density[i];
                  vest[i][1] = v[i][1] + dtfm * f[i][1] * damp + 0.001 * ddx[i][1]/number_density[i];
                  vest[i][2] = v[i][2] + dtfm * f[i][2] * damp + 0.001 * ddx[i][2]/number_density[i];

                  // Compute intermediate (transport) corrected velocity
                  v[i][0] = vest[i][0] - dtfm * ddv[i][0];
	          v[i][1] = vest[i][1] - dtfm * ddv[i][1];
                  v[i][2] = vest[i][2] - dtfm * ddv[i][2];

                  // Update position to n+1
	          x[i][0] += dtv * v[i][0];
                  x[i][1] += dtv * v[i][1];
                  x[i][2] += dtv * v[i][2];

                }
		else { // if solid particle

                  // Compute extrapolated velocity     
                  vest[i][0] = v[i][0] + 2.0 * dtfm * f[i][0] + 0.001 * ddx[i][0]/number_density[i];
                  vest[i][1] = v[i][1] + 2.0 * dtfm * f[i][1] + 0.001 * ddx[i][1]/number_density[i];
                  vest[i][2] = v[i][2] + 2.0 * dtfm * f[i][2] + 0.001 * ddx[i][2]/number_density[i];
                  
		  // Compute half-step velocity
                  v[i][0] += dtfm * f[i][0];
                  v[i][1] += dtfm * f[i][1];
                  v[i][2] += dtfm * f[i][2]; 

                  // damp pseudo-pressure waves
                  vest[i][0] *= dampSolid;
                  vest[i][1] *= dampSolid;
                  vest[i][2] *= dampSolid;
                  v[i][0] *= dampSolid;
                  v[i][1] *= dampSolid;
                  v[i][2] *= dampSolid;

                  // Update position to n+1
                  x[i][0] += dtf * v[i][0];
                  x[i][1] += dtf * v[i][1];
                  x[i][2] += dtf * v[i][2];	    
     
		}

                if (solid_tag[i] == 1) { // if solid particle...
                  for (int m = 0; m < 3; m++) {
                    for (int n = 0; n < 3; n++) {
                      deviatoricTensor[i][m][n] += dtf * ddeviatoricTensor[i][m][n];
                    }
                  }
                }

	    // Compute half-step density
            rhoI[i] = rho[i];

            // Compute extrapolated density
            rho[i] += dtf * drho[i];
            }

            else {  // then solid particles are fixed (velocity = 0, position = x0, acceleration = 0)

                if (solid_tag[i] == 0) { // if fluid particle ...

                    // Compute half-step density
                    rhoI[i] = rho[i];

		    // Compute extrapolated density
		    rho[i] += dtf * drho[i];
                }

                else { // if solid particle ...
                  for (int m = 0; m < 3; m++) {
                    for (int n = 0; n < 3; n++) {
                      deviatoricTensor[i][m][n] += dtf * ddeviatoricTensor[i][m][n];
                    }
                  }
                  // Compute half-step density
                  rhoI[i] = rho[i];           
                }
            }
            
            // Update concentration fields to n+1/2
            dtCm = 0.5*update->dt;
            for (int k = 0; k < atom->num_sdpd_species; k++) {
                C[i][k] += Q[i][k] *dtf;
                C[i][k] = C[i][k] > 0 ? C[i][k] : 0.0;
            }
        }
    }
}

/* ---------------------------------------------------------------------- */

void FixSsaTsdpdBvfFsi::final_integrate() {

    // update v, rho, and e of atoms in group

    double **x = atom->x;
    double **v = atom->v;
    double **f = atom->f;
    double *e = atom->e;
    double *de = atom->de;
    double *rho = atom->rho;
    double *rhoI = atom->rhoI;
    double *drho = atom->drho;
    double dtCm;
    double **C = atom->C;
    double **Q = atom->Q;
    int *type = atom->type;
    int *mask = atom->mask;
    double *mass = atom->mass;
    int nlocal = atom->nlocal;
    if (igroup == atom->firstgroup)
        nlocal = atom->nfirst;
    double dtfm;
    double *rmass = atom->rmass;
    int rmass_flag = atom->rmass_flag;
    int r,ro,k;
    int *solid_tag = atom->solid_tag;
    double *phi = atom->phi;
    double *number_density = atom->number_density;
    double **nw = atom->nw;
    double **v_weighted_solid = atom->v_weighted_solid;
    double **a_weighted_solid = atom->a_weighted_solid;
    double ***ddeviatoricTensor = atom->ddeviatoricTensor;
    double ***deviatoricTensor = atom->deviatoricTensor;
    double **ddx = atom->ddx; 
    double **ddv = atom->ddv;
    double enx,eny,enz,norm_nw,v_dot_en;
    double Ux_solid_nearby, Uy_solid_nearby, Uz_solid_nearby;
    double ax_solid_nearby, ay_solid_nearby, az_solid_nearby;
    double **vest = atom->vest;
    double *rhoAux1 = atom->rhoAux1;
    double *rhoAux2 = atom->rhoAux2;
    int **Cd = atom->Cd;
    int **Qd = atom->Qd;
    int freqFilter = 1e16;
   
    double tnow = (double) update->ntimestep;
    double tmax = (double) update->nsteps;
    double tdamp = 1;
    double damp;
    if (tnow <= tdamp) damp = tnow/tdamp; //*( -0.5 * sin( (-0.5 + tnow/tdamp) * MY_PI ) + 0.5  );
    else damp = 1.0;

    double dampSolid;
    double tdampSolid = 1; //1e6;
    if (tnow <= tdampSolid) dampSolid = 0.0;//tnow/tdampSolid;
    else dampSolid = 1.0;
    

    for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
            if (rmass_flag) {
                dtfm = dtf / rmass[i];
            } else {
                dtfm = dtf / mass[type[i]];
            }


            // Compute phi, nw and averaged velocity and acceleration
            phi[i] = phi[i] / number_density[i];
            nw[i][0] = nw[i][0] / number_density[i];
            nw[i][1] = nw[i][1] / number_density[i];
            nw[i][2] = nw[i][2] / number_density[i];
            
            if (atom -> fixed_tag[i] == 0) { //then particle is free to move

                // Check if fluid particle has phi > 0.5; if so, correct velocity
                if (solid_tag[i] == 0) { // if fluid particle ...                    
                    if (phi[i] > 0.5) { // if phi > 0.5

                        // Get velocity "U" and acceleration "a" of nearby walls (Zhen Li's paper, Eq. 7)
                        Ux_solid_nearby = v_weighted_solid[i][0] / number_density[i];
                        Uy_solid_nearby = v_weighted_solid[i][1] / number_density[i];
                        Uz_solid_nearby = v_weighted_solid[i][2] / number_density[i];
                        ax_solid_nearby = a_weighted_solid[i][0] / number_density[i];
                        ay_solid_nearby = a_weighted_solid[i][1] / number_density[i];
                        az_solid_nearby = a_weighted_solid[i][2] / number_density[i];
                        
                        // Return particle to its original position
                        x[i][0] -= dtv * v[i][0];
                        x[i][1] -= dtv * v[i][1];
                        x[i][2] -= dtv * v[i][2];

                        // Correct the velocity
                        norm_nw = sqrt(nw[i][0]*nw[i][0] + nw[i][1]*nw[i][1] + nw[i][2]*nw[i][2]);
                        enx = -nw[i][0]/norm_nw;
                        eny = -nw[i][1]/norm_nw;
                        enz = -nw[i][2]/norm_nw;
                        v_dot_en =  v[i][0]*enx + v[i][1]*eny + v[i][2]*enz;
                        //v[i][0] = 2.0 * Ux_solid_nearby + ax_solid_nearby * dtf - v[i][0] + 2.0 * std::max(0.0,v_dot_en) * enx;
                        //v[i][1] = 2.0 * Uy_solid_nearby + ay_solid_nearby * dtf - v[i][1] + 2.0 * std::max(0.0,v_dot_en) * eny;
                        //v[i][2] = 2.0 * Uz_solid_nearby + az_solid_nearby * dtf - v[i][2] + 2.0 * std::max(0.0,v_dot_en) * enz;
                        //v[i][0] = 2.0 * Ux_solid_nearby - v[i][0] + 2.0 * std::max(0.0,v_dot_en) * enx;
                        //v[i][1] = 2.0 * Uy_solid_nearby - v[i][1] + 2.0 * std::max(0.0,v_dot_en) * eny;
                        //v[i][2] = 2.0 * Uz_solid_nearby - v[i][2] + 2.0 * std::max(0.0,v_dot_en) * enz;
                        v[i][0] = -v[i][0] + 2.0 * std::max(0.0,v_dot_en) * enx;
                        v[i][1] = -v[i][1] + 2.0 * std::max(0.0,v_dot_en) * eny;
                        v[i][2] = -v[i][2] + 2.0 * std::max(0.0,v_dot_en) * enz;
                        
                        // Use corrected velocity to update particle position
                        x[i][0] += dtv * v[i][0];
                        x[i][1] += dtv * v[i][1];
                        x[i][2] += dtv * v[i][2];
                    }                   
                }
	

                else { // if solid particle...
                  for (int m = 0; m < 3; m++) {
                    for (int n = 0; n < 3; n++) {
                      deviatoricTensor[i][m][n] += dtf * ddeviatoricTensor[i][m][n];
                    }
                  }
	        }	    

                // Compute final velocity
		if (solid_tag[i] == 0) {// if fluid particle ...
                  v[i][0] = vest[i][0] + dtfm * f[i][0] * damp + 0.001 * ddx[i][0]/number_density[i];
                  v[i][1] = vest[i][1] + dtfm * f[i][1] * damp + 0.001 * ddx[i][1]/number_density[i];
                  v[i][2] = vest[i][2] + dtfm * f[i][2] * damp + 0.001 * ddx[i][2]/number_density[i];
                }

		else { //if solid particle ...
                  v[i][0] += dtfm * f[i][0] + 0.001 * ddx[i][0]/number_density[i];
                  v[i][1] += dtfm * f[i][1] + 0.001 * ddx[i][1]/number_density[i];
                  v[i][2] += dtfm * f[i][2] + 0.001 * ddx[i][2]/number_density[i];

                  // damp pseudo-pressure waves
                  v[i][0] *= dampSolid;
                  v[i][1] *= dampSolid;
                  v[i][2] *= dampSolid;

                }


		// Compute final density
		  if (solid_tag[i] == 0){ // if fluid particle ...
                    if ( update->ntimestep % freqFilter == 0) 
                      rho[i] = rhoAux1[i] / rhoAux2[i] + dtf * drho[i];
                    else
                    rho[i] = rhoI[i] + dtv * drho[i];
                  }
                  else{
                    rho[i] = rhoI[i] + dtv * drho[i];
                  }
            }

	    
            else {  // if particle is fixed (velocity = 0, position = x0, acceleration = 0)

                if (solid_tag[i] == 0) { // if fluid particle ...

                    // Compute final density
		    if ( update->ntimestep % freqFilter  == 0) 
                      rho[i] = rhoAux1[i] / rhoAux2[i] + dtv * drho[i];
                    else
                      rho[i] = rhoI[i] + dtv * drho[i]; 
                }

                else { // if solid particle ...
                  for (int m = 0; m < 3; m++) {
                    for (int n = 0; n < 3; n++) {
                      deviatoricTensor[i][m][n] += dtf * ddeviatoricTensor[i][m][n];
                    }
                  }               
		if ( update->ntimestep % freqFilter == 0) 
                  rho[i] = rhoAux1[i] / rhoAux2[i];
                else
                  rho[i] = rhoI[i];
                }

            }
	    

            // Update concentration fields to n+1
            dtCm = 0.5*update->dt;
            for (int k = 0; k < atom->num_sdpd_species; k++) {
                C[i][k] += Q[i][k] *dtf;
                C[i][k] = C[i][k] > 0 ? C[i][k] : 0.0;
            }

	    // SSA diffusion and reactions
	    for (int s=0; s<atom->num_ssa_species; s++){
              Cd[i][s] += Qd[i][s];
	      Cd[i][s] = Cd[i][s] > 0 ? Cd[i][s] : 0;
	    }

            if (atom->num_ssa_species > 0) {
              double tt=0;
              double a0 = 0.0;
              for(r=0;r<atom->num_ssa_reactions;r++) a0 += atom->ssa_rxn_propensity[i][r];
              if(a0 > 0.0){
                double r1 = random->uniform();
                double r2 = random->uniform();
                tt += -log(1.0-r1)/a0;
                double old_a;
                double delta_a0=0.0;
                double a_sum = 0;

                 while(tt < update->dt){
                 //find next reaction to fire
                 for(r=0;r<atom->num_ssa_reactions;r++){
                   if((a_sum += atom->ssa_rxn_propensity[i][r]) > r2*a0) break;
                 }
                 // Change species populations for reaction r
                 for(int s=0;s<atom->num_ssa_species;s++){
                   Cd[i][s] += atom->ssa_stoich_matrix[i][r][s];
                   // Change reaction propensities
                   for(ro=0;ro<atom->num_ssa_reactions;ro++){
                     if(atom->d_ssa_rxn_prop_d_c[i][ro][s] != 0){
                       old_a = atom->ssa_rxn_propensity[i][ro];
                       atom->ssa_rxn_propensity[i][ro] += atom->d_ssa_rxn_prop_d_c[i][ro][s];
                       delta_a0 += (atom->ssa_rxn_propensity[i][ro] - old_a);
                     }
                   }
                 }
                 // Update total propensity
                 a0 += delta_a0;
                 // Roll new random numbers
                 r1 = random->uniform();
                 r2 = random->uniform();
                 // Calculate time to next reaction
                 tt += -log(1.0-r1)/a0;
               }
             }
           }
        }
    }
}


/* ---------------------------------------------------------------------- */

void FixSsaTsdpdBvfFsi::reset_dt() {
    dtv = update->dt;
    dtf = 0.5 * update->dt * force->ftm2v;
}
