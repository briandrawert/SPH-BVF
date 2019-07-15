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
#include "fix_ssa_tsdpd_bvf.h"
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

FixSsaTsdpdBvf::FixSsaTsdpdBvf(LAMMPS *lmp, int narg, char **arg) :
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

int FixSsaTsdpdBvf::setmask() {
    int mask = 0;
    mask |= INITIAL_INTEGRATE;
    mask |= PRE_FORCE;
    mask |= FINAL_INTEGRATE;
    return mask;
}

/* ---------------------------------------------------------------------- */

void FixSsaTsdpdBvf::init() {
    dtv = update->dt;
    dtf = 0.5 * update->dt * force->ftm2v;
}

/* ---------------------------------------------------------------------- */

void FixSsaTsdpdBvf::setup_pre_force(int vflag)
{
    // set vest equal to v
    double **v = atom->v;
    double **vest = atom->vest; 
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    if (igroup == atom->firstgroup)
        nlocal = atom->nfirst;

    for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
            vest[i][0] = v[i][0];
            vest[i][1] = v[i][1];
            vest[i][2] = v[i][2];
        }
    }
}

/* ----------------------------------------------------------------------
 allow for both per-type and per-atom mass
 ------------------------------------------------------------------------- */

void FixSsaTsdpdBvf::initial_integrate(int vflag) {
    // update v and x and rho and e of atoms in group

    double **x = atom->x;
    double **v = atom->v;
    double **f = atom->f;
    double **vest = atom->vest;
    double *rho = atom->rho;
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
    double **nw = atom->nw;
    double **v_weighted_solid = atom->v_weighted_solid;
    double **a_weighted_solid = atom->a_weighted_solid;
    double ***ddeviatoricTensor = atom->ddeviatoricTensor;
    double ***deviatoricTensor = atom->deviatoricTensor;


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
                
                // Estimated velocity (n+1)
                vest[i][0] = v[i][0] + 2.0 * dtfm * f[i][0];
                vest[i][1] = v[i][1] + 2.0 * dtfm * f[i][1];
                vest[i][2] = v[i][2] + 2.0 * dtfm * f[i][2];

                // Update velocity to n+1/2
                v[i][0] += dtfm * f[i][0];
                v[i][1] += dtfm * f[i][1];
                v[i][2] += dtfm * f[i][2];

                // Update position to n+1
                x[i][0] += dtf * v[i][0];
                x[i][1] += dtf * v[i][1];
                x[i][2] += dtf * v[i][2];
	
	
                if (solid_tag[i] == 1) { // if solid particle...
                  for (int m = 0; m < 3; m++) {
                    for (int n = 0; n < 3; n++) {
                      deviatoricTensor[i][m][n] += 0.5 * dtv * ddeviatoricTensor[i][m][n];
                    }
                  }
                }
            }

            else {  // then solid particles are fixed (velocity = 0, position = x0, acceleration = 0)

                if (solid_tag[i] == 0) { // if fluid particle ...

                    // Estimated velocity (n+1)
                    vest[i][0] = v[i][0] + 2.0 * dtfm * f[i][0];
                    vest[i][1] = v[i][1] + 2.0 * dtfm * f[i][1];
                    vest[i][2] = v[i][2] + 2.0 * dtfm * f[i][2];

                    // Update velocity to n+1/2
                    v[i][0] += dtfm * f[i][0];
                    v[i][1] += dtfm * f[i][1];
                    v[i][2] += dtfm * f[i][2];

                    // Update position to n+1/2
                    x[i][0] += dtf * v[i][0];
                    x[i][1] += dtf * v[i][1];
                    x[i][2] += dtf * v[i][2];
                }

                else { // if solid particle ...
                  for (int m = 0; m < 3; m++) {
                    for (int n = 0; n < 3; n++) {
                      deviatoricTensor[i][m][n] += dtf * ddeviatoricTensor[i][m][n];
                    }
                  } 
               }
            }

            // Update concentration fiels to n+1/2
            dtCm = 0.5*update->dt;
            for (int k = 0; k < atom->num_sdpd_species; k++) {
                C[i][k] += Q[i][k] *dtf;
                C[i][k] = C[i][k] > 0 ? C[i][k] : 0.0;
            }

            // Update energy field to n+1
            //e[i] += dtf * de[i]; // half-step update of particle internal energy
 
           // Update density to n+1/2
           rho[i] += dtf * drho[i];
       }
    }
}

/* ---------------------------------------------------------------------- */

void FixSsaTsdpdBvf::final_integrate() {

    // update v, rhom, and e of atoms in group

    double **x = atom->x;
    double **v = atom->v;
    double **f = atom->f;
    double *e = atom->e;
    double *de = atom->de;
    double *rho = atom->rho;
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
    double **nw = atom->nw;
    double **v_weighted_solid = atom->v_weighted_solid;
    double **a_weighted_solid = atom->a_weighted_solid;
    double ***ddeviatoricTensor = atom->ddeviatoricTensor;
    double ***deviatoricTensor = atom->deviatoricTensor;
    double enx,eny,enz,norm_nw,v_dot_en;
    double Ux_solid_nearby, Uy_solid_nearby, Uz_solid_nearby;
    double ax_solid_nearby, ay_solid_nearby, az_solid_nearby;
    
   
    for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
            if (rmass_flag) {
                dtfm = dtf / rmass[i];
            } else {
                dtfm = dtf / mass[type[i]];
            }


            if (atom -> fixed_tag[i] == 0) { //then solid particles are free to move   (velocity = 0, position = x0, acceleration = 0)

                // Update velocity to n+1
                v[i][0] += dtfm * f[i][0];
                v[i][1] += dtfm * f[i][1];
                v[i][2] += dtfm * f[i][2];


                // Check if fluid particle has phi > 0.5; if so, correct velocity
                if (solid_tag[i] == 0) { // if fluid particle ...
                    if (phi[i] > 0.5) { // if phi > 0.5

                        // Get velocity "U" and acceleration "a" of solid walls;(Zhen Li's paper, Eq. 7)
                        Ux_solid_nearby = v_weighted_solid[i][0];
                        Uy_solid_nearby = v_weighted_solid[i][1];
                        Uz_solid_nearby = v_weighted_solid[i][2];
                        ax_solid_nearby = a_weighted_solid[i][0];
                        ay_solid_nearby = a_weighted_solid[i][1];
                        az_solid_nearby = a_weighted_solid[i][2];

                        // Correct the velocity
                        norm_nw = sqrt(nw[i][0]*nw[i][0] + nw[i][1]*nw[i][1] + nw[i][2]*nw[i][2]);
                        enx = -nw[i][0]/norm_nw;
                        eny = -nw[i][1]/norm_nw;
                        enz = -nw[i][2]/norm_nw;
                        v_dot_en =  v[i][0]*enx + v[i][1]*eny + v[i][2]*enz;
                        v[i][0] = 2.0 * Ux_solid_nearby + ax_solid_nearby * dtv - v[i][0] + 2.0 * std::max(0.0,v_dot_en) * enx;
                        v[i][1] = 2.0 * Uy_solid_nearby + ay_solid_nearby * dtv - v[i][1] + 2.0 * std::max(0.0,v_dot_en) * eny;
                        v[i][2] = 2.0 * Uz_solid_nearby + az_solid_nearby * dtv - v[i][2] + 2.0 * std::max(0.0,v_dot_en) * enz;
                    }
                }

                else { // if solid particle...
                  for (int m = 0; m < 3; m++) {
                    for (int n = 0; n < 3; n++) {
                      deviatoricTensor[i][m][n] += dtf * ddeviatoricTensor[i][m][n];
                    }
                  }
                }

            }

            else {  // then solid particles are fixed (velocity = 0, position = x0, acceleration = 0)

                if (solid_tag[i] == 0) { // if fluid particle ...

                    // Update velocity to n+1
                    v[i][0] += dtfm * f[i][0];
                    v[i][1] += dtfm * f[i][1];
                    v[i][2] += dtfm * f[i][2];

                    // Update position to n+1
                    x[i][0] += dtf * v[i][0];
                    x[i][1] += dtf * v[i][1];
                    x[i][2] += dtf * v[i][2];
	    
                    // Check if fluid particle has phi > 0.5; if so, correct velocity
                    if (phi[i] > 0.5) { // if phi > 0.5
                        // Nearby solid velocity and acceleration is U = a = 0 (Zhen Li's paper, Eq. 7)

                        // Correct the velocity
                        norm_nw = sqrt(nw[i][0]*nw[i][0] + nw[i][1]*nw[i][1] + nw[i][2]*nw[i][2]);
                        enx = -nw[i][0]/norm_nw;
                        eny = -nw[i][1]/norm_nw;
                        enz = -nw[i][2]/norm_nw;
                        v_dot_en =  v[i][0]*enx + v[i][1]*eny + v[i][2]*enz;
                        v[i][0] = - v[i][0] + 2.0 * std::max(0.0,v_dot_en) * enx;
                        v[i][1] = - v[i][1] + 2.0 * std::max(0.0,v_dot_en) * eny;
                        v[i][2] = - v[i][2] + 2.0 * std::max(0.0,v_dot_en) * enz;
                    }
                }

                else { // if solid particle ...
                  for (int m = 0; m < 3; m++) {
                    for (int n = 0; n < 3; n++) {
                      deviatoricTensor[i][m][n] += dtf * ddeviatoricTensor[i][m][n];
                    }
                  }                
               }
            }

            // Update concentration fields to n+1
            dtCm = 0.5*update->dt;
            for (int k = 0; k < atom->num_sdpd_species; k++) {
                C[i][k] += Q[i][k] *dtf;
                C[i][k] = C[i][k] > 0 ? C[i][k] : 0.0;
            }

            // Update energy field to n+1
            //e[i] += dtf * de[i]; // half-step update of particle internal energy
            
            // Update density to n+1
            rho[i] += dtf * drho[i];	
        }
    }
}


/* ---------------------------------------------------------------------- */

void FixSsaTsdpdBvf::reset_dt() {
    dtv = update->dt;
    dtf = 0.5 * update->dt * force->ftm2v;
}
