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
#include "fix_ssa_tsdpd_bvf_mechanics.h"
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

FixSsaTsdpdBvfMechanics::FixSsaTsdpdBvfMechanics(LAMMPS *lmp, int narg, char **arg) :
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

int FixSsaTsdpdBvfMechanics::setmask() {
    int mask = 0;
    mask |= INITIAL_INTEGRATE;
    mask |= PRE_FORCE;
    mask |= FINAL_INTEGRATE;
    return mask;
}

/* ---------------------------------------------------------------------- */

void FixSsaTsdpdBvfMechanics::init() {
    dtv = update->dt;
    dtf = 0.5 * update->dt * force->ftm2v;
}

/* ---------------------------------------------------------------------- */

void FixSsaTsdpdBvfMechanics::setup_pre_force(int vflag)
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

/* ---------------------------------------------------------------------- */
 void FixSsaTsdpdBvfMechanics::initial_integrate(int vflag) {

   double ** x = atom -> x;
   double ** v = atom -> v;
   double ** f = atom -> f;
   double ** vest = atom -> vest;
   double * rho = atom -> rho;
   double * rhoI = atom -> rhoI;
   double * drho = atom -> drho;
   double * e = atom -> e;
   double * de = atom -> de;
   double * mass = atom -> mass;
   double * rmass = atom -> rmass;
   int rmass_flag = atom -> rmass_flag;
   double dtCm;
   double ** C = atom -> C;
   double ** Q = atom -> Q;
   int * type = atom -> type;
   int * mask = atom -> mask;
   int nlocal = atom -> nlocal;
   int i, j;
   double dtfm;
   double enx, eny, enz, norm_nw, v_dot_en;
   double Ux_solid_nearby, Uy_solid_nearby, Uz_solid_nearby;
   double ax_solid_nearby, ay_solid_nearby, az_solid_nearby;
   int * solid_tag = atom -> solid_tag;
   double * phi = atom -> phi;
   double * number_density = atom -> number_density;
   double ** nw = atom -> nw;
   double ** v_weighted_solid = atom -> v_weighted_solid;
   double ** a_weighted_solid = atom -> a_weighted_solid;
   double *** ddeviatoricTensor = atom -> ddeviatoricTensor;
   double *** deviatoricTensor = atom -> deviatoricTensor;
   double ** ddv = atom -> ddv;
   double ** ddx = atom -> ddx;
   double * rhoAux1 = atom -> rhoAux1;
   double * rhoAux2 = atom -> rhoAux2;
   double tnow = (double) update -> ntimestep;
   double tmax = (double) update -> nsteps;
   double damp, dampSolid, tdamp, tdampSolid;
   double smoothingFactor;
   
   // Smoothing factor for velocity
   smoothingFactor = 0.001;

   // Compute damping factor as a function of time step (for fluid particles)
   tdamp = 1;
   if (tnow <= tdamp) damp = tnow / tdamp; //*( -0.5 * sin( (-0.5 + tnow/tdamp) * MY_PI ) + 0.5  );
   else damp = 1.0;

   // Compute damping factor as a function of time step (for solid particles)
   tdampSolid = 1e6; // release solid at time step 1e6
   if (tnow < tdampSolid) dampSolid = 0.0; //tnow/tdampSolid;
   else dampSolid = 1.0;

   if (igroup == atom -> firstgroup)
     nlocal = atom -> nfirst;

   for (i = 0; i < nlocal; i++) {
     if (mask[i] & groupbit) {
       if (rmass_flag) {
         dtfm = dtf / rmass[i];
       } else {
         dtfm = dtf / mass[type[i]];
       }

       // If particles are free to move
       if (atom -> fixed_tag[i] == 0) {

         // If fluid particle
         if (solid_tag[i] == 0) {

           // Compute intermediate (momentum) velocity     
           vest[i][0] = v[i][0] + dtfm * f[i][0] * damp + smoothingFactor * ddx[i][0] / number_density[i];
           vest[i][1] = v[i][1] + dtfm * f[i][1] * damp + smoothingFactor * ddx[i][1] / number_density[i];
           vest[i][2] = v[i][2] + dtfm * f[i][2] * damp + smoothingFactor * ddx[i][2] / number_density[i];

           // Compute intermediate (transport) corrected velocity
           v[i][0] = vest[i][0] - dtfm * ddv[i][0];
           v[i][1] = vest[i][1] - dtfm * ddv[i][1];
           v[i][2] = vest[i][2] - dtfm * ddv[i][2];

           // Update position to n+1
           x[i][0] += dtv * v[i][0];
           x[i][1] += dtv * v[i][1];
           x[i][2] += dtv * v[i][2];
         }

         // If solid particle
         else {

           // Compute extrapolated velocity     
           vest[i][0] = v[i][0] + 2.0 * dtfm * f[i][0] + smoothingFactor * ddx[i][0] / number_density[i];
           vest[i][1] = v[i][1] + 2.0 * dtfm * f[i][1] + smoothingFactor * ddx[i][1] / number_density[i];
           vest[i][2] = v[i][2] + 2.0 * dtfm * f[i][2] + smoothingFactor * ddx[i][2] / number_density[i];

           // Compute half-step velocity
           v[i][0] += dtfm * f[i][0];
           v[i][1] += dtfm * f[i][1];
           v[i][2] += dtfm * f[i][2];

           // Damp pseudo-pressure waves
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

           // Compute half-step deviatoric tensor
           for (int m = 0; m < 3; m++) {
             for (int n = 0; n < 3; n++) {
               deviatoricTensor[i][m][n] += dtf * ddeviatoricTensor[i][m][n];
             }
           }
         }

         // For both fluid and solid particles

         // Compute half-step density
         rhoI[i] = rho[i];

         // Compute extrapolated density
         rho[i] += dtf * drho[i];
       }

       // If particles are fixed (vel = 0, position = x0, acceleration = 0)
       else {

         // If fluid particle
         if (solid_tag[i] == 0) {

           // Compute half-step density
           rhoI[i] = rho[i];

           // Compute extrapolated density
           rho[i] += dtf * drho[i];
         }

         // If solid particle
         else {
           for (int m = 0; m < 3; m++) {
             for (int n = 0; n < 3; n++) {
               deviatoricTensor[i][m][n] += dtf * ddeviatoricTensor[i][m][n];
             }
           }
           // Compute half-step density
           rhoI[i] = rho[i];
         }
       }

       // For all particles

       // Update concentration fields to n+1/2
       dtCm = 0.5 * update -> dt;
       for (int k = 0; k < atom -> num_sdpd_species; k++) {
         C[i][k] += Q[i][k] * dtf;
         C[i][k] = C[i][k] > 0 ? C[i][k] : 0.0;
       }
     }
   }
 }

 /* ---------------------------------------------------------------------- */

 void FixSsaTsdpdBvfMechanics::final_integrate() {

   double ** x = atom -> x;
   double ** v = atom -> v;
   double ** f = atom -> f;
   double * e = atom -> e;
   double * de = atom -> de;
   double * rho = atom -> rho;
   double * rhoI = atom -> rhoI;
   double * drho = atom -> drho;
   double dtCm;
   double ** C = atom -> C;
   double ** Q = atom -> Q;
   int * type = atom -> type;
   int * mask = atom -> mask;
   double * mass = atom -> mass;
   int nlocal = atom -> nlocal;
   if (igroup == atom -> firstgroup)
     nlocal = atom -> nfirst;
   double dtfm;
   double * rmass = atom -> rmass;
   int rmass_flag = atom -> rmass_flag;
   int r, ro, k;
   int * solid_tag = atom -> solid_tag;
   double * phi = atom -> phi;
   double * number_density = atom -> number_density;
   double ** nw = atom -> nw;
   double ** v_weighted_solid = atom -> v_weighted_solid;
   double ** a_weighted_solid = atom -> a_weighted_solid;
   double *** ddeviatoricTensor = atom -> ddeviatoricTensor;
   double *** deviatoricTensor = atom -> deviatoricTensor;
   double ** ddx = atom -> ddx;
   double ** ddv = atom -> ddv;
   double enx, eny, enz, norm_nw, v_dot_en;
   double Ux_solid_nearby, Uy_solid_nearby, Uz_solid_nearby;
   double ax_solid_nearby, ay_solid_nearby, az_solid_nearby;
   double ** vest = atom -> vest;
   double * rhoAux1 = atom -> rhoAux1;
   double * rhoAux2 = atom -> rhoAux2;
   int ** Cd = atom -> Cd;
   int ** Qd = atom -> Qd;
   int freqFilter = 20;
   double tnow = (double) update -> ntimestep;
   double tmax = (double) update -> nsteps;
   double damp, dampSolid, tdamp, tdampSolid;
   double smoothingFactor;
   
   // Smoothing factor for velocity
   smoothingFactor = 0.001;

   // Compute damping factor as a function of time step (for fluid particles)
   tdamp = 1;
   if (tnow <= tdamp) damp = tnow / tdamp; //*( -0.5 * sin( (-0.5 + tnow/tdamp) * MY_PI ) + 0.5  );
   else damp = 1.0;

   // Compute damping factor as a function of time step (for solid particles)
   dampSolid;
   tdampSolid = 1e6; //release solid at time step 1e6
   if (tnow < tdampSolid) dampSolid = 0.0; //tnow/tdampSolid;
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

       // If particle is free to move 
       if (atom -> fixed_tag[i] == 0) {

         // If fluid particle
         if (solid_tag[i] == 0) {

           // Check if fluid particle has phi > 0.5; if so, correct velocity (BVF)
           if (phi[i] > 0.5) {

             // Get velocity "U" and acceleration "a" of nearby walls
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
             norm_nw = sqrt(nw[i][0] * nw[i][0] + nw[i][1] * nw[i][1] + nw[i][2] * nw[i][2]);
             enx = -nw[i][0] / norm_nw;
             eny = -nw[i][1] / norm_nw;
             enz = -nw[i][2] / norm_nw;
             v_dot_en = v[i][0] * enx + v[i][1] * eny + v[i][2] * enz;
             //v[i][0] = 2.0 * Ux_solid_nearby + ax_solid_nearby * dtf - v[i][0] + 2.0 * std::max(0.0,v_dot_en) * enx;
             //v[i][1] = 2.0 * Uy_solid_nearby + ay_solid_nearby * dtf - v[i][1] + 2.0 * std::max(0.0,v_dot_en) * eny;
             //v[i][2] = 2.0 * Uz_solid_nearby + az_solid_nearby * dtf - v[i][2] + 2.0 * std::max(0.0,v_dot_en) * enz;
             v[i][0] = -v[i][0] + 2.0 * std::max(0.0, v_dot_en) * enx;
             v[i][1] = -v[i][1] + 2.0 * std::max(0.0, v_dot_en) * eny;
             v[i][2] = -v[i][2] + 2.0 * std::max(0.0, v_dot_en) * enz;

             // Use corrected velocity to update particle position
             x[i][0] += dtv * v[i][0];
             x[i][1] += dtv * v[i][1];
             x[i][2] += dtv * v[i][2];

           }

           // Compute final velocity
           v[i][0] = vest[i][0] + dtfm * f[i][0] * damp + smoothingFactor * ddx[i][0] / number_density[i];
           v[i][1] = vest[i][1] + dtfm * f[i][1] * damp + smoothingFactor * ddx[i][1] / number_density[i];
           v[i][2] = vest[i][2] + dtfm * f[i][2] * damp + smoothingFactor * ddx[i][2] / number_density[i];

           // Compute final density
           if (update -> ntimestep % freqFilter == 0)
             rho[i] = rhoAux1[i] / rhoAux2[i] + dtf * drho[i];
           else
             rho[i] = rhoI[i] + dtv * drho[i];

         }

         // If solid particle
         else {

           // Compute final velocity
           v[i][0] += dtfm * f[i][0] + smoothingFactor * ddx[i][0] / number_density[i];
           v[i][1] += dtfm * f[i][1] + smoothingFactor * ddx[i][1] / number_density[i];
           v[i][2] += dtfm * f[i][2] + smoothingFactor * ddx[i][2] / number_density[i];

           // Damp pseudo-pressure waves
           v[i][0] *= dampSolid;
           v[i][1] *= dampSolid;
           v[i][2] *= dampSolid;

           // Compute final step of deviatoric stress tensor
           for (int m = 0; m < 3; m++) {
             for (int n = 0; n < 3; n++) {
               deviatoricTensor[i][m][n] += dtf * ddeviatoricTensor[i][m][n];
             }
           }
           // Compute final density
           rho[i] = rhoI[i] + dtv * drho[i];

         }
       }

       // If particle is fixed (velocity = 0, position = x0, acceleration = 0)
       else {

         // If fluid particle
         if (solid_tag[i] == 0) {

           // Compute final density
           if (update -> ntimestep % freqFilter == 0)
             rho[i] = rhoAux1[i] / rhoAux2[i] + dtv * drho[i];
           else
             rho[i] = rhoI[i] + dtv * drho[i];
         }

         // If solid particle
         else {
           for (int m = 0; m < 3; m++) {
             for (int n = 0; n < 3; n++) {
               deviatoricTensor[i][m][n] += dtf * ddeviatoricTensor[i][m][n];
             }
           }
           if (update -> ntimestep % freqFilter == 0)
             rho[i] = rhoAux1[i] / rhoAux2[i];
           else
             rho[i] = rhoI[i];
         }
       }

       // For all particles
       
       // Update concentration fields to n+1
       dtCm = 0.5 * update -> dt;
       for (int k = 0; k < atom -> num_sdpd_species; k++) {
         C[i][k] += Q[i][k] * dtf;
         C[i][k] = C[i][k] > 0 ? C[i][k] : 0.0;
       }

       // SSA diffusion and reactions
       for (int s = 0; s < atom -> num_ssa_species; s++) {
         Cd[i][s] += Qd[i][s];
         Cd[i][s] = Cd[i][s] > 0 ? Cd[i][s] : 0;
       }

       if (atom -> num_ssa_species > 0) {
         double tt = 0;
         double a0 = 0.0;
         for (r = 0; r < atom -> num_ssa_reactions; r++) a0 += atom -> ssa_rxn_propensity[i][r];
         if (a0 > 0.0) {
           double r1 = random -> uniform();
           double r2 = random -> uniform();
           tt += -log(1.0 - r1) / a0;
           double old_a;
           double delta_a0 = 0.0;
           double a_sum = 0;

           while (tt < update -> dt) {
             //find next reaction to fire
             for (r = 0; r < atom -> num_ssa_reactions; r++) {
               if ((a_sum += atom -> ssa_rxn_propensity[i][r]) > r2 * a0) break;
             }
             // Change species populations for reaction r
             for (int s = 0; s < atom -> num_ssa_species; s++) {
               Cd[i][s] += atom -> ssa_stoich_matrix[i][r][s];
               // Change reaction propensities
               for (ro = 0; ro < atom -> num_ssa_reactions; ro++) {
                 if (atom -> d_ssa_rxn_prop_d_c[i][ro][s] != 0) {
                   old_a = atom -> ssa_rxn_propensity[i][ro];
                   atom -> ssa_rxn_propensity[i][ro] += atom -> d_ssa_rxn_prop_d_c[i][ro][s];
                   delta_a0 += (atom -> ssa_rxn_propensity[i][ro] - old_a);
                 }
               }
             }
             // Update total propensity
             a0 += delta_a0;
             // Roll new random numbers
             r1 = random -> uniform();
             r2 = random -> uniform();
             // Calculate time to next reaction
             tt += -log(1.0 - r1) / a0;
           }
         }
       }
     }
   }
 }

/* ---------------------------------------------------------------------- */

void FixSsaTsdpdBvfMechanics::reset_dt() {
    dtv = update->dt;
    dtf = 0.5 * update->dt * force->ftm2v;
}
