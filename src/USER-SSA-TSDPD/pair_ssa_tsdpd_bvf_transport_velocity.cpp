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

#include <math.h>
#include <stdlib.h> 
#include "pair_ssa_tsdpd_bvf_transport_velocity.h"
#include "atom.h"
#include "force.h"
#include "comm.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"
#include "domain.h"
#include "update.h"
#include "random_mars.h"
#include <set> 
#include <unistd.h> 
#include <time.h> 
#include "string.h"
#include <stdio.h> 
#include <time.h>
#include "math_const.h"

  using namespace LAMMPS_NS;
  using namespace MathConst;

/* ---------------------------------------------------------------------- */

PairSsaTsdpdBvfTransportVelocity::PairSsaTsdpdBvfTransportVelocity(LAMMPS * lmp): Pair(lmp) {
  restartinfo = 0;
  first = 1;
  random = NULL;
}

/* ---------------------------------------------------------------------- */

PairSsaTsdpdBvfTransportVelocity::~PairSsaTsdpdBvfTransportVelocity() {
  if (allocated) {
    memory -> destroy(setflag);
    memory -> destroy(cutsq);
    memory -> destroy(cut);
    memory -> destroy(rho0);
    memory -> destroy(soundspeed);
    memory -> destroy(B);
    memory -> destroy(viscosity);
    memory -> destroy(kappa);
    memory -> destroy(cutc);
    memory -> destroy(strainTensor);
    memory -> destroy(rotationTensor);
    memory -> destroy(kroneckerTensor);
    memory -> destroy(transportTensor);
    memory -> destroy(G0);
    if (atom -> num_ssa_species > 0) memory -> destroy(kappaSSA);
  }
  if (random) delete random;
}

void PairSsaTsdpdBvfTransportVelocity::compute(int eflag, int vflag) {
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, fpair, pij;
  double randnum;
  int * ilist, * jlist, * numneigh, ** firstneigh;
  double vxtmp, vytmp, vztmp, imass, jmass, fi, fj, fvisc, fviscs, h, ih, ihsq, velx, vely, velz;
  double ftransportx, ftransporty, ftransportz;
  double rsq, tmp, wfd, wf, wfBvf, delVdotDelR, deltaE, mu;

  //  if (atom->ssa_diffusion_flag == 1) std::set < int > * dfsp_D_matrix_index = new std::set < int > [atom->nmax]; // set for each column to store which elements are non-zero

  if (eflag || vflag)
    ev_setup(eflag, vflag);
  else
    evflag = vflag_fdotr = 0;

  double ** vt = atom -> v;
  double ** v = atom -> vest;
  double ** x = atom -> x;
  double ** f = atom -> f;
  double * rho = atom -> rho;
  double * rhoI = atom -> rhoI;
  double * rhoAux1 = atom -> rhoAux1;
  double * rhoAux2 = atom -> rhoAux2;
  double * mass = atom -> mass;
  double * de = atom -> de;
  double * e = atom -> e;
  double * drho = atom -> drho;
  double ** C = atom -> C;
  double ** Q = atom -> Q;
  int * type = atom -> type;
  int nlocal = atom -> nlocal;
  int newton_pair = force -> newton_pair;
  int dimension = domain -> dimension;
  double kBoltzmann = force -> boltz;
  double dtinv = 1.0 / update -> dt;
  int nmax = atom -> nmax;
  int * solid_tag = atom -> solid_tag;
  double * phi = atom -> phi;
  double * number_density = atom -> number_density;
  double ** nw = atom -> nw;
  double ** v_weighted_solid = atom -> v_weighted_solid;
  double ** a_weighted_solid = atom -> a_weighted_solid;
  double ** * ddeviatoricTensor = atom -> ddeviatoricTensor;
  double ** * deviatoricTensor = atom -> deviatoricTensor;
  double ** * artificialStressTensor = atom -> artificialStressTensor;
  double deviatoricDotRotation, rotationDotDeviatoric;
  double xDotDeviatoricx, xDotDeviatoricy, xDotDeviatoricz;
  double ** ddx = atom -> ddx;
  double ** ddv = atom -> ddv;
  double Pi, Pj, Pij;
  double wfd2, wdelta, delta;
  double fr;
  double xDotArtificialStressx, xDotArtificialStressy, xDotArtificialStressz;
  double totalStressi, totalStressj;

  if (first) {
    for (i = 1; i <= atom -> ntypes; i++) {
      for (j = 1; i <= atom -> ntypes; i++) {
        if (cutsq[i][j] > 1.e-32) {
          if (!setflag[i][i] || !setflag[j][j]) {
            if (comm -> me == 0) {
              printf(
                "SsaTsdpd particle types %d and %d interact with cutoff=%g, but not all of their single particle properties are set.\n",
                i, j, sqrt(cutsq[i][j]));
            }
          }
        }
      }
    }
    first = 0;
  }

  inum = list -> inum;
  ilist = list -> ilist;
  numneigh = list -> numneigh;
  firstneigh = list -> firstneigh;
  int inumsq = inum * inum;

  // Declare variables and allocate local arrays in case of SSA simulation
  int ** Cd = atom -> Cd;
  int ** Qd = atom -> Qd;
  double * dfsp_D_matrix = atom -> dfsp_D_matrix;
  double * dfsp_D_diag = atom -> dfsp_D_diag;
  // double * dfsp_Diffusion_coeff = atom -> dfsp_Diffusion_coeff;
  double * dfsp_a_i = atom -> dfsp_a_i;
  std::set < int > * dfsp_D_matrix_index;
  if (atom -> ssa_diffusion_flag == 1) dfsp_D_matrix_index = new std::set < int > [atom -> nmax]; // set for each column to store which elements are non-zero

  // Compute Kronecker delta tensor
  for (int m = 0; m < 3; m++) {
    for (int n = 0; n < 3; n++) {
      if (m == n) kroneckerTensor[m][n] = 1.0;
      else kroneckerTensor[m][n] = 0.0;
    }
  }

  // ==================================================================
  // compute number density, normals
  // for solid mechanics: compute rate of strain and rotation tensors
  // ==================================================================

  for (ii = 0; ii < inum; ii++) {

    i = ilist[ii];

    //printf("\t\ti=%i\n",i);
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    vxtmp = v[i][0];
    vytmp = v[i][1];
    vztmp = v[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    imass = mass[itype];
    Pi = 7.0 * B[itype] * (rho[i] / rho0[itype] - 1.0);
    double hRatio = 1.0;

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;
      jtype = type[j];
      jmass = mass[jtype];
      Pj = 7.0 * B[jtype] * (rho[j] / rho0[jtype] - 1.0);

      if (rsq < cutsq[itype][jtype] * pow(hRatio, 2)) {
        h = cut[itype][jtype]; // for Lucy kernel
        double r = sqrt(rsq);

        if (domain -> dimension == 3) { // Kernel, 3d (1/r * dwdr)
          //Lucy kernel (3D)
          ih = 1.0 / h;
          ihsq = ih * ih;
          wfd = h - sqrt(rsq);
          wfd = -25.066903536973515383e0 * wfd * wfd * ihsq * ihsq * ihsq * ih;
          wf = h - sqrt(rsq);
          wf = 2.088908628081126 * wf * wf * wf * ihsq * ihsq * ihsq * ih * (h + 3. * r);
          wfBvf = h - sqrt(rsq);
          wfBvf = 2.088908628081126 * wfBvf * wfBvf * wfBvf * ihsq * ihsq * ihsq * ih * (h + 3. * r);
          wfd2 = hRatio * h - sqrt(rsq);
          wfd2 = -25.066903536973515383e0 * wfd2 * wfd2 * ihsq * ihsq * ihsq * ih / pow(hRatio, 7);

        } else if (domain -> dimension == 2) { // Kernel, 2d (1/r * dwdr)
          //Lucy kernel (2D)
          ih = 1.0 / h;
          ihsq = ih * ih;
          wfd = h - sqrt(rsq);
          wfd = -19.098593171027440292e0 * wfd * wfd * ihsq * ihsq * ihsq;
          wf = h - sqrt(rsq);
          wf = 1.591549430918954 * wf * wf * wf * ihsq * ihsq * ihsq * (h + 3. * r);
          wfBvf = h - sqrt(rsq);
          wfBvf = 1.591549430918954 * wfBvf * wfBvf * wfBvf * ihsq * ihsq * ihsq * (h + 3. * r);
          wfd2 = hRatio * h - sqrt(rsq);
          wfd2 = -19.098593171027440292e0 * wfd2 * wfd2 * ihsq * ihsq * ihsq / pow(hRatio, 6);

        } else if (domain -> dimension == 1) { // Kernel, 1d (1/r * dwdr)
          //Lucy kernel (1D)
          ih = 1.0 / h;
          ihsq = ih * ih;
          wfd = h - sqrt(rsq);
          wfd = -15.0 * wfd * wfd * ihsq * ihsq * ih;
          wf = 1. - r * ih;
          wf = (5. / 4.) * ih * (wf * wf * wf) * (1. + 3. * r * ih);
          wfd2 = hRatio * h - sqrt(rsq);
          wfd2 = -15.0 * wfd2 * wfd2 * ihsq * ihsq * ih / pow(hRatio, 5);

        }

        //number density of particles (Li et al., 2018, BVF paper, Eq. 2)
        number_density[i] += pow(jmass / rho[j], 2) * wfBvf;

        // rhoAux
        double rhoAux = rhoI[j];
        rhoAux1[i] += rhoAux * wfBvf;
        rhoAux2[i] += wfBvf;
       
        //Velocity correction (Adami et al., J Comp Phys 241 (2013))
        ddv[i][0] += 10.0 * 7.0 * B[itype] * ((imass / rho[i]) * (imass / rho[i]) + (jmass / rho[j]) * (jmass / rho[j])) * wfd2 * delx; //(Adami et al., J Comp Phys 241 (2013))
        ddv[i][1] += 10.0 * 7.0 * B[itype] * ((imass / rho[i]) * (imass / rho[i]) + (jmass / rho[j]) * (jmass / rho[j])) * wfd2 * dely; //(Adami et al., J Comp Phys 241 (2013))
        ddv[i][2] += 10.0 * 7.0 * B[itype] * ((imass / rho[i]) * (imass / rho[i]) + (jmass / rho[j]) * (jmass / rho[j])) * wfd2 * delz; //(Adami et al., J Comp Phys 241 (2013))

        // Reactions in neighbors (j particles)
        if (newton_pair || j < nlocal) {

          //number density of particles (Li et al., 2018, BVF paper, Eq. 2)
          number_density[j] += pow(imass / rho[i], 2) * wfBvf;

          // rhoAux
          double rhoAux = rhoI[i];
          rhoAux1[j] += rhoAux * wfBvf;
          rhoAux2[j] += wfBvf;

          //Velocity correction (Adami et al., J Comp Phys 241 (2013))
          ddv[j][0] += 10.0 * 7.0 * B[jtype] * ((jmass / rho[j]) * (jmass / rho[j]) + (imass / rho[i]) * (imass / rho[i])) * wfd2 * (-delx); //(Adami et al., J Comp Phys 241 (2013))
          ddv[j][1] += 10.0 * 7.0 * B[jtype] * ((jmass / rho[j]) * (jmass / rho[j]) + (imass / rho[i]) * (imass / rho[i])) * wfd2 * (-dely); //(Adami et al., J Comp Phys 241 (2013))
          ddv[j][2] += 10.0 * 7.0 * B[jtype] * ((jmass / rho[j]) * (jmass / rho[j]) + (imass / rho[i]) * (imass / rho[i])) * wfd2 * (-delz); //(Adami et al., J Comp Phys 241 (2013))

        }
      }
    }
  }

  // ==================================================================
  // main loop
  // ==================================================================
  //printf("\tStarting i loop\n");
  for (ii = 0; ii < inum; ii++) {

    i = ilist[ii];

    //printf("\t\ti=%i\n",i);
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    vxtmp = v[i][0];
    vytmp = v[i][1];
    vztmp = v[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    imass = mass[itype];

    // compute pressure of atom i with Tait EOS
    tmp = rho[i] / rho0[itype];
    fi = 7.0 * B[itype] * (tmp - 1.0);

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];

      rsq = delx * delx + dely * dely + delz * delz;
      jtype = type[j];
      jmass = mass[jtype];

      if (rsq < cutsq[itype][jtype]) {
        h = cut[itype][jtype]; // for Lucy kernel
        double r = sqrt(rsq);
        delta = (1.0 / 2.6) * h;

        if (domain -> dimension == 3) { // Kernel, 3d (1/r * dwdr)
          //Lucy kernel (3D)
          ih = 1.0 / h;
          ihsq = ih * ih;
          wfd = h - sqrt(rsq);
          wfd = -25.066903536973515383e0 * wfd * wfd * ihsq * ihsq * ihsq * ih;
          wf = h - sqrt(rsq);
          wf = 2.088908628081126 * wf * wf * wf * ihsq * ihsq * ihsq * ih * (h + 3. * r);
          wfBvf = h - sqrt(rsq);
          wfBvf = 2.088908628081126 * wfBvf * wfBvf * wfBvf * ihsq * ihsq * ihsq * ih * (h + 3. * r);
          wdelta = h - delta;
          wdelta = 2.088908628081126 * wdelta * wdelta * wdelta * ihsq * ihsq * ihsq * ih * (h + 3. * delta);

        } else if (domain -> dimension == 2) { // Kernel, 2d (1/r * dwdr)
          //Lucy kernel (2D)
          ih = 1.0 / h;
          ihsq = ih * ih;
          wfd = h - sqrt(rsq);
          wfd = -19.098593171027440292e0 * wfd * wfd * ihsq * ihsq * ihsq;
          wf = h - sqrt(rsq);
          wf = 1.591549430918954 * wf * wf * wf * ihsq * ihsq * ihsq * (h + 3. * r);
          wfBvf = h - sqrt(rsq);
          wfBvf = 1.591549430918954 * wfBvf * wfBvf * wfBvf * ihsq * ihsq * ihsq * (h + 3. * r);
          wdelta = h - delta;
          wdelta = 1.591549430918954 * wdelta * wdelta * wdelta * ihsq * ihsq * ihsq * (h + 3. * delta);

        } else if (domain -> dimension == 1) { // Kernel, 1d (1/r * dwdr)
          //Lucy kernel (1D)
          ih = 1.0 / h;
          ihsq = ih * ih;
          wfd = h - sqrt(rsq);
          wfd = -15.0 * wfd * wfd * ihsq * ihsq * ih; //Lucy (1d)
          wf = 1. - r * ih;
          wf = (5. / 4.) * ih * (wf * wf * wf) * (1. + 3. * r * ih);
          wdelta = 1. - delta * ih;
          wdelta = (5. / 4.) * ih * (wdelta * wdelta * wdelta) * (1. + 3. * delta * ih);

        }

        // compute pressure  of atom j with Tait EOS
        tmp = rho[j] / rho0[jtype];
        fj = 7.0 * B[jtype] * (tmp - 1.0);

        // velocity differences
        velx = vxtmp - v[j][0];
        vely = vytmp - v[j][1];
        velz = vztmp - v[j][2];

        // dot product of velocity delta and distance vector
        delVdotDelR = delx * velx + dely * vely + delz * velz;

        // transport force and transport tensor (relevant only for inviscid flows)
        for (int m = 0; m < 3; m++) {
          for (int n = 0; n < 3; n++) {
            transportTensor[m][n] = 0.5 * ((rho[i] * v[i][m] * (vt[i][n] - v[i][n])) + (rho[j] * v[j][m] * (vt[j][n] - v[j][n])));
          }
        }
        ftransportx = ( pow(imass/rho[i],2) + pow(jmass/rho[j],2) ) * (transportTensor[0][0] * delx + transportTensor[0][1] * dely + transportTensor[0][2] * delz) * wfd;
        ftransporty = ( pow(imass/rho[i],2) + pow(jmass/rho[j],2) ) * (transportTensor[1][0] * delx + transportTensor[1][1] * dely + transportTensor[1][2] * delz) * wfd;
        ftransportz = ( pow(imass/rho[i],2) + pow(jmass/rho[j],2) ) * (transportTensor[2][0] * delx + transportTensor[2][1] * dely + transportTensor[2][2] * delz) * wfd;

        // Artificial viscosity (Managhan, 1992)
        //if (delVdotDelR < 0.) {
        //fvisc = - imass * jmass * 8.0 * ((double) domain->dimension + 2.0) * viscosity[itype][jtype] * delVdotDelR * wfd / ( (rsq + 0.01*h*h) * pow((rho[i]+rho[j]),2) );
        //} else {
        // fvisc = 0.;
        //}

        // Artificial Viscosity (Adami et al., 2013)
        fvisc = (pow(imass / rho[i], 2) + pow(jmass / rho[j], 2)) * (viscosity[itype][jtype] * wfd);

        // Pressure force (Adami et al., 2013, with pressure switch by Sun et al., 2018)
        //if (fi + fj >= 0.) pij = ((rho[i] * fj + rho[j] * fi) / (rho[i] + rho[j])); 
        //else               pij = ((rho[i] * fj - rho[j] * fi) / (rho[i] + rho[j]));  
	//fpair = (pow(imass / rho[i], 2) + pow(jmass / rho[j], 2)) * pij * wfd;  // (Adami, 2010)
	//if (solid_tag[i] == 1 && solid_tag[j] == 1) fpair = (pow(imass / rho[i], 2) + pow(jmass / rho[j], 2)) * ((rho[i] * fj + rho[j] * fi) / (rho[i] + rho[j])) * wfd; 

        // Pressure force (Zhang et al., 2017, with pressure switch by Sun et al., 2018)
	pij = ( (fj/(rho[j]*rho[j])) + (fi/(rho[i]*rho[i])) );
	if (pij >= 0.) fpair = imass * jmass * ( (fj/(rho[j]*rho[j])) + (fi/(rho[i]*rho[i])) ) * wfd;
	else           fpair = imass * jmass * ( (fj/(rho[j]*rho[j])) - (fi/(rho[i]*rho[i])) ) * wfd;
        if (solid_tag[i] == 1 && solid_tag[j] == 1) fpair = imass * jmass * ( (fj/(rho[j]*rho[j])) + (fi/(rho[i]*rho[i])) ) * wfd;


        //Repulsive force evaluation (Monaghan, 2013, Eq. 2.14) (Just relevant for multiphase flows)
        fr = 0.0; //0.08 * imass * jmass * fabs( (rho0[itype] - rho0[jtype]) / (rho0[itype] + rho0[jtype]) ) * fabs( (fi + fj) / (rho[i]*rho[j]) ) * wfd;


        // Random force calculation
        double wiener[3][3] = {
          0
        };
        for (int l = 0; l < dimension; l++) {
          for (int m = 0; m < dimension; m++) {
            wiener[l][m] = random -> gaussian();
          }
        }

        // symmetric part
        wiener[0][1] = wiener[1][0] = (wiener[0][1] + wiener[1][0]) / 2.;
        wiener[0][2] = wiener[2][0] = (wiener[0][2] + wiener[2][0]) / 2.;
        wiener[1][2] = wiener[2][1] = (wiener[1][2] + wiener[2][1]) / 2.;

        // traceless part
        double trace_over_dim = (wiener[0][0] + wiener[1][1] + wiener[2][2]) / dimension;
        wiener[0][0] -= trace_over_dim;
        wiener[1][1] -= trace_over_dim;
        wiener[2][2] -= trace_over_dim;

        double prefactor = sqrt(-4. * kBoltzmann * e[i] * (imass * jmass * wfd / (rho[i] * rho[j])) * dtinv) / (r + 0.01 * h);
        double f_random[3] = {
          0
        };
        for (int l = 0; l < dimension; ++l) f_random[l] = prefactor * (wiener[l][0] * delx + wiener[l][1] * dely + wiener[l][2] * delz);

        // Compute deviatoric stress tensor for particle i
        // (1) Compute rate of strain and rotation tensors
        for (int m = 0; m < 3; m++) {
          for (int n = 0; n < 3; n++) {
            strainTensor[m][n] = 0.5 * (jmass / rho[j]) * wfd * (((v[j][m] - v[i][m]) * (x[i][n] - x[j][n])) + ((v[j][n] - v[i][n]) * (x[i][m] - x[j][m])));
            rotationTensor[m][n] = 0.5 * (jmass / rho[j]) * wfd * (((v[j][m] - v[i][m]) * (x[i][n] - x[j][n])) - ((v[j][n] - v[i][n]) * (x[i][m] - x[j][m])));
          }
        }

        // (2) Compute rate of change of Jaumann's rate equation
        if (solid_tag[i] == 1) { // if i is a solid particle
          for (int m = 0; m < 3; m++) {
            for (int n = 0; n < 3; n++) {
              deviatoricDotRotation = deviatoricTensor[i][m][0] * rotationTensor[n][0] + deviatoricTensor[i][m][1] * rotationTensor[n][1] + deviatoricTensor[i][m][2] * rotationTensor[n][2];
              rotationDotDeviatoric = rotationTensor[m][0] * deviatoricTensor[i][0][n] + rotationTensor[m][1] * deviatoricTensor[i][1][n] + rotationTensor[m][2] * deviatoricTensor[i][2][n];
              ddeviatoricTensor[i][m][n] += 2.0 * ((2.0 * G0[itype] * G0[jtype]) / (G0[itype] + G0[jtype] + 1e-12)) * (strainTensor[m][n] - (1. / 3.) * kroneckerTensor[m][n] * strainTensor[m][n]) + deviatoricDotRotation + rotationDotDeviatoric;
            }
          }
        }

        // Compute artificial stress tensor for particle i
        if (solid_tag[i] == 1) { // if i is a solid particle
          for (int m = 0; m < 3; m++) {
            for (int n = 0; n < 3; n++) {
              totalStressi = deviatoricTensor[i][m][n] - fi * kroneckerTensor[m][n];
              if (totalStressi > 0.0) artificialStressTensor[i][m][n] = -0.35 * totalStressi / (rho[i] * rho[i]);
              else artificialStressTensor[i][m][n] = 0.0;
            }
          }
        } else { // if i is a fluid particle
          for (int m = 0; m < 3; m++) {
            if (fi < 0.0) artificialStressTensor[i][m][m] = 0.0 * fabs(fi) / (rho[i] * rho[i]);
            else artificialStressTensor[i][m][m] = 0.0 * (fi / (rho[i] * rho[i]));
          }
        }

        // Compute artificial stress tensor for particle j
        if (solid_tag[j] == 1) { // if j is a solid particle
          for (int m = 0; m < 3; m++) {
            for (int n = 0; n < 3; n++) {
              totalStressj = deviatoricTensor[j][m][n] - fj * kroneckerTensor[m][n];
              if (totalStressj > 0.0) artificialStressTensor[j][m][n] = -0.35 * totalStressj / (rho[j] * rho[j]);
              else artificialStressTensor[j][m][n] = 0.0;
            }
          }
        } else { // if j is a fluid particle
          for (int m = 0; m < 3; m++) {
            if (fj < 0.0) artificialStressTensor[j][m][m] = 0.0 * fabs(fj) / (rho[j] * rho[j]);
            else artificialStressTensor[j][m][m] = 0.0 * (fj / (rho[j] * rho[j]));
          }
        }

        // compute x dot artificialStressTensor
        xDotArtificialStressx = imass * jmass * wfd * pow(wf / wdelta, 4) * (delx * (artificialStressTensor[i][0][0] + artificialStressTensor[j][0][0]) +
          dely * (artificialStressTensor[i][1][0] + artificialStressTensor[j][1][0]) +
          delz * (artificialStressTensor[i][2][0] + artificialStressTensor[j][2][0]));
        xDotArtificialStressy = imass * jmass * wfd * pow(wf / wdelta, 4) * (delx * (artificialStressTensor[i][0][1] + artificialStressTensor[j][0][1]) +
          dely * (artificialStressTensor[i][1][1] + artificialStressTensor[j][1][1]) +
          delz * (artificialStressTensor[i][2][1] + artificialStressTensor[j][2][1]));
        xDotArtificialStressz = imass * jmass * wfd * pow(wf / wdelta, 4) * (delx * (artificialStressTensor[i][0][2] + artificialStressTensor[j][0][2]) +
          dely * (artificialStressTensor[i][1][2] + artificialStressTensor[j][1][2]) +
          delz * (artificialStressTensor[i][2][2] + artificialStressTensor[j][2][2]));

        //Momentum evaluation
        if (solid_tag[i] == 0) { // if i is a fluid particle ...
          // Accumulate total force for particle i
	  f[i][0] += -delx * fpair + fvisc * velx + f_random[0] + delx * fr + ftransportx + xDotArtificialStressx;
          f[i][1] += -dely * fpair + fvisc * vely + f_random[1] + dely * fr + ftransporty + xDotArtificialStressy;
          f[i][2] += -delz * fpair + fvisc * velz + f_random[2] + delz * fr + ftransportz + xDotArtificialStressz;


        } else { // i is a solid particle
          // add deviatoric stress component
          xDotDeviatoricx = imass * jmass * wfd * (delx * (deviatoricTensor[i][0][0] / (rho[i] * rho[i]) + deviatoricTensor[j][0][0] / (rho[j] * rho[j])) +
            dely * (deviatoricTensor[i][1][0] / (rho[i] * rho[i]) + deviatoricTensor[j][1][0] / (rho[j] * rho[j])) +
            delz * (deviatoricTensor[i][2][0] / (rho[i] * rho[i]) + deviatoricTensor[j][2][0] / (rho[j] * rho[j])));
          xDotDeviatoricy = imass * jmass * wfd * (delx * (deviatoricTensor[i][0][1] / (rho[i] * rho[i]) + deviatoricTensor[j][0][1] / (rho[j] * rho[j])) +
            dely * (deviatoricTensor[i][1][1] / (rho[i] * rho[i]) + deviatoricTensor[j][1][1] / (rho[j] * rho[j])) +
            delz * (deviatoricTensor[i][2][1] / (rho[i] * rho[i]) + deviatoricTensor[j][2][1] / (rho[j] * rho[j])));
          xDotDeviatoricz = imass * jmass * wfd * (delx * (deviatoricTensor[i][0][2] / (rho[i] * rho[i]) + deviatoricTensor[j][0][2] / (rho[j] * rho[j])) +
            dely * (deviatoricTensor[i][1][2] / (rho[i] * rho[i]) + deviatoricTensor[j][1][2] / (rho[j] * rho[j])) +
            delz * (deviatoricTensor[i][2][2] / (rho[i] * rho[i]) + deviatoricTensor[j][2][2] / (rho[j] * rho[j])));

          // artificial viscosity for solids (Pereira et al., 2017)
          if (delVdotDelR < 0.) {
            mu = h * delVdotDelR / (rsq + 0.01 * h * h);
            fviscs = imass * jmass * wfd * (-(soundspeed[itype] + soundspeed[jtype]) * mu + 2.0 * mu * mu) / (rho[i] + rho[j]);
          } else {
            fviscs = 0.;
          }

          // Accumulate total force for particle i
          f[i][0] += -delx * fpair - delx * fviscs + xDotDeviatoricx + delx * fr + xDotArtificialStressx;
          f[i][1] += -dely * fpair - dely * fviscs + xDotDeviatoricy + dely * fr + xDotArtificialStressy;
          f[i][2] += -delz * fpair - delz * fviscs + xDotDeviatoricz + delz * fr + xDotArtificialStressz;

        }


        // damp pseudo-sound waves travelling through the domain at the beginning of the simulation (high-frequency oscillations)
        double tnow = update->ntimestep * update->dt;
        double tmax = update->dt * update->nsteps;
        double tdamp = tmax;
        double damp;
        double amplDamp = 0.0;
        if (tnow <= tdamp) damp = amplDamp; //*( -0.5 * sin( (-0.5 + tnow/tdamp) * MY_PI ) + 0.5  );
        else damp = 0.0;

        //classical density formulation
        //drho[i] += rho[i] * jmass * delVdotDelR * wfd / rho[j];

        //artificial density diffusion: Molteni (2009)
        //drho[i] += rho[i] * jmass * delVdotDelR * wfd / rho[j] - 0.05 * h * rho[i] * soundspeed[itype] * jmass * 2.0 * (rho[j] / rho[i] - 1.0) * (rsq / (rsq + 0.01 * h * h)) * wfd / rho[j];

        //new density formulation
        double velxt = vt[i][0] - vt[j][0];
        double velyt = vt[i][1] - vt[j][1];
        double velzt = vt[i][2] - vt[j][2];
        double delVtdotDelR = delx * velxt + dely * velyt + delz * velzt;
        drho[i] += ( rho[i] * jmass * delVtdotDelR * wfd / rho[j] ) 
		   - damp *  h * rho[i] * soundspeed[itype] * jmass * 2.0 * (rho[j] / rho[i] - 1.0) * (rsq / (rsq + 0.01 * h * h)) * wfd / rho[j]
		   - (jmass/rho[j]) * (rho[i] * ( (v[i][0]-vt[i][0])*delx + (v[i][1]-vt[i][1])*dely + (v[i][2]-vt[i][2])*delz )
		                    +  rho[j] * ( (v[j][0]-vt[j][0])*delx + (v[j][1]-vt[j][1])*dely + (v[j][2]-vt[j][2])*delz ) ) * wfd;

        //Energy evaluation
        deltaE = -0.5 * (fpair * delVdotDelR + fvisc * (velx * velx + vely * vely + velz * velz));
        de[i] += deltaE;


        // Compute phi (boundary volume fraction)
        if (solid_tag[i] == 0) { // if i is a fluid particle ...
          if (solid_tag[j] == 1) { // if j is a solid particle ...
            phi[i] += pow(jmass / rho[j], 2) * wfBvf;
          }
        }

        // Compute normals
        if (solid_tag[i] == 0) { // if i is a fluid particle ...
          if (solid_tag[j] == 1) { // if j is a solid particle ...
            nw[i][0] += delx * wfd * pow(jmass / rho[j], 2);
            nw[i][1] += dely * wfd * pow(jmass / rho[j], 2);
            nw[i][2] += delz * wfd * pow(jmass / rho[j], 2);
          }
        }

        // Reactions in neighbors (j particles)
        if (newton_pair || j < nlocal) {

          //Momentum evaluation

          // Compute deviatoric stress tensor for particle j
          // (1) Compute rate of strain and rotation tensors
          for (int m = 0; m < 3; m++) {
            for (int n = 0; n < 3; n++) {
              strainTensor[m][n] = 0.5 * (imass / rho[i]) * wfd * (((v[i][m] - v[j][m]) * (x[j][n] - x[i][n])) + ((v[i][n] - v[j][n]) * (x[j][m] - x[i][m])));
              rotationTensor[m][n] = 0.5 * (imass / rho[i]) * wfd * (((v[i][m] - v[j][m]) * (x[j][n] - x[i][n])) - ((v[i][n] - v[j][n]) * (x[j][m] - x[i][m])));
            }
          }

          // (2) Compute rate of change of Jaumann's rate equation
          if (solid_tag[j] == 1) { // if j is a solid particle
            for (int m = 0; m < 3; m++) {
              for (int n = 0; n < 3; n++) {
                deviatoricDotRotation = deviatoricTensor[j][m][0] * rotationTensor[n][0] + deviatoricTensor[j][m][1] * rotationTensor[n][1] + deviatoricTensor[j][m][2] * rotationTensor[n][2];
                rotationDotDeviatoric = rotationTensor[m][0] * deviatoricTensor[j][0][n] + rotationTensor[m][1] * deviatoricTensor[j][1][n] + rotationTensor[m][2] * deviatoricTensor[j][2][n];
                ddeviatoricTensor[j][m][n] += 2.0 * ((2.0 * G0[itype] * G0[jtype]) / (G0[itype] + G0[jtype] + 1e-14)) * (strainTensor[m][n] - (1. / 3.) * kroneckerTensor[m][n] * strainTensor[m][n]) + deviatoricDotRotation + rotationDotDeviatoric;
              }
            }
          }

          if (solid_tag[j] == 0) { // if j is a fluid particle ...

            // Accumulate total force for particle j
	    if (pij < 0.) fpair = -fpair;
            f[j][0] -= (-delx * fpair + fvisc * velx + f_random[0] + delx * fr + ftransportx + xDotArtificialStressx);
            f[j][1] -= (-dely * fpair + fvisc * vely + f_random[1] + dely * fr + ftransporty + xDotArtificialStressy);
            f[j][2] -= (-delz * fpair + fvisc * velz + f_random[2] + delz * fr + ftransportz + xDotArtificialStressz);

          } else { // j is a solid particle

            // add deviatoric stress component
            xDotDeviatoricx = imass * jmass * wfd * (delx * (deviatoricTensor[j][0][0] / (rho[j] * rho[j]) + deviatoricTensor[i][0][0] / (rho[i] * rho[i])) +
              dely * (deviatoricTensor[j][1][0] / (rho[j] * rho[j]) + deviatoricTensor[i][1][0] / (rho[i] * rho[i])) +
              delz * (deviatoricTensor[j][2][0] / (rho[j] * rho[j]) + deviatoricTensor[i][2][0] / (rho[i] * rho[i])));
            xDotDeviatoricy = imass * jmass * wfd * (delx * (deviatoricTensor[j][0][1] / (rho[j] * rho[j]) + deviatoricTensor[i][0][1] / (rho[i] * rho[i])) +
              dely * (deviatoricTensor[j][1][1] / (rho[j] * rho[j]) + deviatoricTensor[i][1][1] / (rho[i] * rho[i])) +
              delz * (deviatoricTensor[j][2][1] / (rho[j] * rho[j]) + deviatoricTensor[i][2][1] / (rho[i] * rho[i])));
            xDotDeviatoricz = imass * jmass * wfd * (delx * (deviatoricTensor[j][0][2] / (rho[j] * rho[j]) + deviatoricTensor[i][0][2] / (rho[i] * rho[i])) +
              dely * (deviatoricTensor[j][1][2] / (rho[j] * rho[j]) + deviatoricTensor[i][1][2] / (rho[i] * rho[i])) +
              delz * (deviatoricTensor[j][2][2] / (rho[j] * rho[j]) + deviatoricTensor[i][2][2] / (rho[i] * rho[i])));

            // artificial viscosity for solids (Pereira et al., 2017)
            if (delVdotDelR < 0.) {
              mu = h * delVdotDelR / (rsq + 0.01 * h * h);
              fviscs = imass * jmass * wfd * (-(soundspeed[itype] + soundspeed[jtype]) * mu + 2.0 * mu * mu) / (rho[i] + rho[j]);
            } else {
              fviscs = 0.;
            }

            // Accumulate total force for particle j
            f[j][0] -= (-delx * fpair - delx * fviscs + xDotDeviatoricx + delx * fr + xDotArtificialStressx);
            f[j][1] -= (-dely * fpair - dely * fviscs + xDotDeviatoricy + dely * fr + xDotArtificialStressy);
            f[j][2] -= (-delz * fpair - delz * fviscs + xDotDeviatoricz + delz * fr + xDotArtificialStressz);

          }

          //Density evaluation

          //classical density formulation
          //drho[j] += rho[j] * imass * delVdotDelR * wfd / rho[i];

          //artificial density diffusion: Molteni (2009)
          //drho[j] += rho[j] * imass * delVdotDelR * wfd / rho[i] - 0.05 * h * rho[j] * soundspeed[jtype] * imass * 2.0 * (rho[i] / rho[j] - 1.0) * (rsq / (rsq + 0.01 * h * h)) * wfd / rho[i];

          //new density formulation
          double velxt = vt[i][0] - vt[j][0];
          double velyt = vt[i][1] - vt[j][1];
          double velzt = vt[i][2] - vt[j][2];
          double delVtdotDelR = delx * velxt + dely * velyt + delz * velzt; 
          drho[j] += ( rho[j] * imass * delVtdotDelR * wfd / rho[i] ) 
	  	      - damp * h * rho[j] * soundspeed[jtype] * imass * 2.0 * (rho[i] / rho[j] - 1.0) * (rsq / (rsq + 0.01 * h * h)) * wfd / rho[i]
	  	      + (imass/rho[i]) * (rho[j] * ( (v[j][0]-vt[j][0])*delx + (v[j][1]-vt[j][1])*dely + (v[j][2]-vt[j][2])*delz ) +
                                          rho[i] * ( (v[i][0]-vt[i][0])*delx + (v[i][1]-vt[i][1])*dely + (v[i][2]-vt[i][2])*delz ) ) * wfd;
	               
          //Energy evaluation
          de[j] += deltaE;

          // Compute phi (boundary volume fraction)
          if (solid_tag[j] == 0) { // if j is a fluid particle ...
            if (solid_tag[i] == 1) { // if i is a solid particle ...
              phi[j] += pow(imass / rho[i], 2) * wfBvf;
            }
          }

          // Compute normals
          if (solid_tag[j] == 0) { // if j is a fluid particle ...
            if (solid_tag[i] == 1) { // if i is a solid particle ...
              nw[j][0] += -delx * wfd * pow(imass / rho[i], 2);
              nw[j][1] += -dely * wfd * pow(imass / rho[i], 2);
              nw[j][2] += -delz * wfd * pow(imass / rho[i], 2);
            }
          }
        }

        // transport of species
        if (r < cutc[itype][jtype]) {

          double r = sqrt(rsq);
          h = cutc[itype][jtype];

          if (domain -> dimension == 3) { // Kernel, 3d (1/r * dwdr)
            //Lucy kernel (3D)
            ih = 1.0 / h;
            ihsq = ih * ih;
            wfd = h - sqrt(rsq);
            wfd = -25.066903536973515383e0 * wfd * wfd * ihsq * ihsq * ihsq * ih;
            wf = h - sqrt(rsq);
            wf = 2.088908628081126 * wf * wf * wf * ihsq * ihsq * ihsq * ih * (h + 3. * r);

          } else if (domain -> dimension == 2) { // Kernel, 2d (1/r * dwdr)
            //Lucy kernel (2D)
            ih = 1.0 / h;
            ihsq = ih * ih;
            wfd = h - sqrt(rsq);
            wfd = -19.098593171027440292e0 * wfd * wfd * ihsq * ihsq * ihsq;
            wf = h - sqrt(rsq);
            wf = 1.591549430918954 * wf * wf * wf * ihsq * ihsq * ihsq * (h + 3. * r);

          } else if (domain -> dimension == 1) { // Kernel, 1d (1/r * dwdr)
            //Lucy kernel (1D)
            ih = 1.0 / h;
            ihsq = ih * ih;
            wfd = h - sqrt(rsq);
            wfd = -15.0 * wfd * wfd * ihsq * ihsq * ih; //Lucy (1d)
            wf = 1. - r * ih;
            wf = (5. / 4.) * ih * (wf * wf * wf) * (1. + 3. * r * ih);

          }

          double dQc_base = 2.0 * ((imass * jmass) / (imass + jmass)) * ((rho[i] + rho[j]) / (rho[i] * rho[j])) * (delx * delx + dely * dely + delz * delz) * wfd / (rsq + 0.01 * h * h); // (Tartakovsky et. al., 2007, JCP)
          for (int k = 0; k < atom -> num_sdpd_species; ++k) {
            Q[i][k] += (kappa[itype][jtype][k]) * (C[i][k] - C[j][k]) * dQc_base - (jmass/rho[j]) * (C[i][k] * ( (v[i][0]-vt[i][0])*delx + (v[i][1]-vt[i][1])*dely + (v[i][2]-vt[i][2])*delz )
                                                                                                  +  C[j][k] * ( (v[j][0]-vt[j][0])*delx + (v[j][1]-vt[j][1])*dely + (v[j][2]-vt[j][2])*delz ) ) * wfd;
            if (newton_pair || j < nlocal) {
              Q[j][k] -= ( (kappa[itype][jtype][k]) * (C[i][k] - C[j][k]) * dQc_base - (imass/rho[i]) * (C[i][k] * ( (v[i][0]-vt[i][0])*delx + (v[i][1]-vt[i][1])*dely + (v[i][2]-vt[i][2])*delz )
                                                                                                      +  C[j][k] * ( (v[j][0]-vt[j][0])*delx + (v[j][1]-vt[j][1])*dely + (v[j][2]-vt[j][2])*delz ) ) * wfd );
            }
          }
          if (atom -> ssa_diffusion_flag == 1) {
            dfsp_D_matrix[i * inum + j] = -dQc_base;
            dfsp_D_matrix_index[i].insert(j);
            dfsp_D_matrix[j * inum + i] = -dQc_base;
            dfsp_D_matrix_index[j].insert(i);
            //for (int s = 0; s < atom -> num_ssa_species; s++) {
            //dfsp_Diffusion_coeff[i * inum + j + s * inumsq] = kappaSSA[itype][jtype][s];
            //dfsp_Diffusion_coeff[j * inum + i + s * inumsq] = kappaSSA[jtype][itype][s];
            //}
          }
        }

        if (evflag)
          ev_tally(i, j, nlocal, newton_pair, 0.0, 0.0, fpair, delx, dely, delz);
      }
    }
  }

  // Calculate SSA Diffusion
  // Find Diagional element values
  if (atom -> num_ssa_species > 0) {
    std::set < int > ::iterator it;
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      double total = 0;
      for (it = dfsp_D_matrix_index[i].begin(); it != dfsp_D_matrix_index[i].end(); ++it) { // Using a set::iterator
        total += * it;
      }
      dfsp_D_diag[i] = total;
    }

    double tt, a0, sum_d, sum_d2, r1, r2, r3, a_i_src_orig, a_i_dest_orig;
    int k;
    for (int s = 0; s < atom -> num_ssa_species; s++) { // Calculate each species seperatly
      // sum each voxel propensity to get total propensity
      a0 = 0;
      for (ii = 0; ii < inum; ii++) {
        i = ilist[ii];
        dfsp_a_i[i] = 0;
        for (it = dfsp_D_matrix_index[i].begin(); it != dfsp_D_matrix_index[i].end(); ++it) { // iterator over outbound connections
          j = * it;
          //dfsp_a_i[i] += dfsp_Diffusion_coeff[i * inum + j + s * inumsq] * dfsp_D_matrix[i * inum + j]; // Note, "dfsp_a_i" is the per-voxel base propensity (must multiply Cd[i][s]);
          dfsp_a_i[i] += kappaSSA[itype][jtype][s] * dfsp_D_matrix[i * inum + j]; // Note, "dfsp_a_i" is the per-voxel base propensity (must multiply Cd[i][s]);
          //        printf("dfsp_Diffusion_coeff = %.16f \n",dfsp_Diffusion_coeff[i*inum+j+s*inumsq]);
        }
        a0 += dfsp_a_i[i] * Cd[i][s];
      }
      // Find time to first reaction
      tt = 0;
      r1 = random -> uniform();
      tt += -log(1.0 - r1) / a0;
      // Loop over time
      while (tt < update -> dt) {
        // find which voxel the diffusion event occured
        r2 = a0 * random -> uniform();
        sum_d = 0;
        for (k = 0; k < inum; k++) {
          sum_d += dfsp_a_i[k] * Cd[k][s];
          if (sum_d > r2) break;
          //printf("\t k=%i sum_d=%e r2=%e\n",k,sum_d,r2);
        }
        int src_vox = k;
        // find which voxel it moved to
        r3 = dfsp_a_i[k] * random -> uniform();
        sum_d2 = 0;
        for (it = dfsp_D_matrix_index[src_vox].begin(); it != dfsp_D_matrix_index[src_vox].end(); ++it) {
          j = * it;
          //sum_d2 += dfsp_Diffusion_coeff[src_vox * inum + j + s * inumsq] * dfsp_D_matrix[src_vox * inum + j];
          sum_d2 += kappaSSA[type[src_vox]][jtype][s] * dfsp_D_matrix[src_vox * inum + j];
          if (sum_d2 > r3) break;
        }
        int dest_vox = j;
        //Cd[src_vox][s]--;
        //Cd[dest_vox][s]++;
        Qd[src_vox][s]--;
        Qd[dest_vox][s]++;
        // Find delta in propensities
        a0 += (dfsp_a_i[dest_vox] - dfsp_a_i[src_vox]);

        //printf("SSA_Diffusion[s=%i] tt=%e %i -> %i.  Now a0=%e",s,tt,src_vox,dest_vox,a0);
        // Find time to next reaction
        r1 = random -> uniform();
        tt += -log(1.0 - r1) / (a0);
        //printf(" tt=%e\n",tt);
      }
    }
    // Dealocate the arrays
    delete[] dfsp_D_matrix_index;
  }

  // =====================================================================
  // compute weighted velocity and acceleration of nearby solid particles
  // =====================================================================

  for (ii = 0; ii < inum; ii++) {

    i = ilist[ii];

    //printf("\t\ti=%i\n",i);
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    vxtmp = v[i][0];
    vytmp = v[i][1];
    vztmp = v[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    imass = mass[itype];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;
      jtype = type[j];
      jmass = mass[jtype];

      if (rsq < cutsq[itype][jtype]) {
        h = cut[itype][jtype]; // for Lucy kernel
        double r = sqrt(rsq);

        if (domain -> dimension == 3) { // Kernel, 3d (1/r * dwdr)
          //Lucy kernel (3D)
          ih = 1.0 / h;
          ihsq = ih * ih;
          wfd = h - sqrt(rsq);
          wfd = -25.066903536973515383e0 * wfd * wfd * ihsq * ihsq * ihsq * ih;
          wf = h - sqrt(rsq);
          wf = 2.088908628081126 * wf * wf * wf * ihsq * ihsq * ihsq * ih * (h + 3. * r);
          wfBvf = h - sqrt(rsq);
          wfBvf = 2.088908628081126 * wfBvf * wfBvf * wfBvf * ihsq * ihsq * ihsq * ih * (h + 3. * r);

        } else if (domain -> dimension == 2) { // Kernel, 2d (1/r * dwdr)
          //Lucy kernel (2D)
          ih = 1.0 / h;
          ihsq = ih * ih;
          wfd = h - sqrt(rsq);
          wfd = -19.098593171027440292e0 * wfd * wfd * ihsq * ihsq * ihsq;
          wf = h - sqrt(rsq);
          wf = 1.591549430918954 * wf * wf * wf * ihsq * ihsq * ihsq * (h + 3. * r);
          wfBvf = h - sqrt(rsq);
          wfBvf = 1.591549430918954 * wfBvf * wfBvf * wfBvf * ihsq * ihsq * ihsq * (h + 3. * r);

        } else if (domain -> dimension == 1) { // Kernel, 1d (1/r * dwdr)
          //Lucy kernel (1D)
          ih = 1.0 / h;
          ihsq = ih * ih;
          wfd = h - sqrt(rsq);
          wfd = -15.0 * wfd * wfd * ihsq * ihsq * ih; //Lucy (1d)
          wf = 1. - r * ih;
          wf = (5. / 4.) * ih * (wf * wf * wf) * (1. + 3. * r * ih);
        }

        // Compute weighted velocity and acceleration of nearby solid particles
        if (solid_tag[i] == 0) { // if i is a fluid particle ...
          if (solid_tag[j] == 1) { // if j is a solid particle ...
            v_weighted_solid[i][0] += v[j][0] * wfBvf * pow(jmass / rho[j], 2);
            v_weighted_solid[i][1] += v[j][1] * wfBvf * pow(jmass / rho[j], 2);
            v_weighted_solid[i][2] += v[j][2] * wfBvf * pow(jmass / rho[j], 2);
            a_weighted_solid[i][0] += (f[j][0] / jmass) * wfBvf * pow(jmass / rho[j], 2);
            a_weighted_solid[i][1] += (f[j][1] / jmass) * wfBvf * pow(jmass / rho[j], 2);
            a_weighted_solid[i][2] += (f[j][2] / jmass) * wfBvf * pow(jmass / rho[j], 2);
          }
        }

        // Reactions in neighbors (j particles)
        if (newton_pair || j < nlocal) {

          // Compute weighted velocity and acceleration of nearby solid particles
          if (solid_tag[j] == 0) { // if j is a fluid particle ...
            if (solid_tag[i] == 1) { // if i is a solid particle ...
              v_weighted_solid[j][0] += v[i][0] * wfBvf * pow(imass / rho[i], 2);
              v_weighted_solid[j][1] += v[i][1] * wfBvf * pow(imass / rho[i], 2);
              v_weighted_solid[j][2] += v[i][2] * wfBvf * pow(imass / rho[i], 2);
              a_weighted_solid[j][0] += (f[i][0] / imass) * wfBvf * pow(imass / rho[i], 2);
              a_weighted_solid[j][1] += (f[i][1] / imass) * wfBvf * pow(imass / rho[i], 2);
              a_weighted_solid[j][2] += (f[i][2] / imass) * wfBvf * pow(imass / rho[i], 2);
            }
          }
        }
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();

}

/* ----------------------------------------------------------------------
 allocate all arrays
 ------------------------------------------------------------------------- */

void PairSsaTsdpdBvfTransportVelocity::allocate() {
  allocated = 1;
  int n = atom -> ntypes;

  memory -> create(setflag, n + 1, n + 1, "pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory -> create(cutsq, n + 1, n + 1, "pair:cutsq");
  memory -> create(rho0, n + 1, "pair:rho0");
  memory -> create(soundspeed, n + 1, "pair:soundspeed");
  memory -> create(B, n + 1, "pair:B");
  memory -> create(cut, n + 1, n + 1, "pair:cut");
  memory -> create(viscosity, n + 1, n + 1, "pair:viscosity");
  memory -> create(kappa, n + 1, n + 1, atom -> num_sdpd_species, "pair:kappa");
  memory -> create(cutc, n + 1, n + 1, "pair:cutc");
  memory -> create(strainTensor, 3, 3, "pair:strainTensor");
  memory -> create(rotationTensor, 3, 3, "pair:rotationTensor");
  memory -> create(kroneckerTensor, 3, 3, "pair:kroneckerTensor");
  memory -> create(transportTensor, 3, 3, "pair:transportTensor");
  memory -> create(G0, n + 1, "pair:G0");

  if (atom -> num_ssa_species > 0) memory -> create(kappaSSA, n + 1, n + 1, atom -> num_ssa_species, "pair:kappaSSA");

}

/* ----------------------------------------------------------------------
   global settings
 ------------------------------------------------------------------------- */

void PairSsaTsdpdBvfTransportVelocity::settings(int narg, char ** arg) {
  if (narg != 0)
    error -> all(FLERR,
      "Illegal number of setting arguments for pair_style ssa_tsdpd/bvf");

  // seed is immune to underflow/overflow because it is unsigned
  //  seed = comm->nprocs + comm->me + atom->nlocal;
  //  if (narg == 3) seed += force->inumeric (FLERR, arg[2]);
  //  random = new RanMars (lmp, seed);

  srand(clock());
  seed = comm -> nprocs + comm -> me + atom -> nlocal + rand() % 100;
  random = new RanMars(lmp, seed);

}

/* ----------------------------------------------------------------------
 set coeffs for one or more type pairs
 ------------------------------------------------------------------------- */

void PairSsaTsdpdBvfTransportVelocity::coeff(int narg, char ** arg) {
  if (narg < 7 + atom -> num_sdpd_species)
    error -> all(FLERR, "Incorrect args for pair_style ssa_tsdpd/bvf coefficients");
  if (!allocated)
    allocate();

  int ilo, ihi, jlo, jhi;
  force -> bounds(FLERR, arg[0], atom -> ntypes, ilo, ihi);
  force -> bounds(FLERR, arg[1], atom -> ntypes, jlo, jhi);

  double rho0_one = force -> numeric(FLERR, arg[2]);
  double soundspeed_one = force -> numeric(FLERR, arg[3]);
  double viscosity_one = force -> numeric(FLERR, arg[4]);
  double cut_one = force -> numeric(FLERR, arg[5]);
  double B_one = soundspeed_one * soundspeed_one * rho0_one / 7.0;
  double cutc_one = force -> numeric(FLERR, arg[6]);
  double G0_one = force -> numeric(FLERR, arg[7]);

  // Read diffusivity
  double kappa_one[atom -> num_sdpd_species];
  for (int k = 0; k < atom -> num_sdpd_species; k++) {
    kappa_one[k] = atof(arg[8 + k]);
  }

  int arg_index = 8 + atom -> num_sdpd_species;

  // Read SSA diffusivity
  double kappaSSA_one[atom -> num_ssa_species];
  for (int k = 0; k < atom -> num_ssa_species; k++) {
    kappaSSA_one[k] = atof(arg[arg_index + k]);
  }

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    rho0[i] = rho0_one;
    soundspeed[i] = soundspeed_one;
    B[i] = B_one;
    G0[i] = G0_one;

    for (int j = MAX(jlo, i); j <= jhi; j++) {
      viscosity[i][j] = viscosity_one;
      cut[i][j] = cut_one;
      cutc[i][j] = cutc_one;

      for (int k = 0; k < atom -> num_sdpd_species; k++) {
        kappa[i][j][k] = kappa_one[k];
      }

      for (int k = 0; k < atom -> num_ssa_species; k++) {
        kappaSSA[i][j][k] = kappaSSA_one[k];
      }

      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0)
    error -> all(FLERR, "Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
 init for one type pair i,j and corresponding j,i
 ------------------------------------------------------------------------- */

double PairSsaTsdpdBvfTransportVelocity::init_one(int i, int j) {

  if (setflag[i][j] == 0) {
    error -> all(FLERR, "Not all pair ssa_tsdpd/bvf coeffs are not set");
  }

  cut[j][i] = cut[i][j];
  viscosity[j][i] = viscosity[i][j];

  for (int k = 0; k < atom -> num_sdpd_species; k++) {
    kappa[j][i][k] = kappa[i][j][k];
  }

  for (int k = 0; k < atom -> num_ssa_species; k++) {
    kappaSSA[j][i][k] = kappaSSA[i][j][k];
  }

  cutc[j][i] = cutc[i][j];

  return cut[i][j];
}

/* ---------------------------------------------------------------------- */

double PairSsaTsdpdBvfTransportVelocity::single(int i, int j, int itype, int jtype,
  double rsq, double factor_coul, double factor_lj, double & fforce) {
  fforce = 0.0;

  return 0.0;
}
