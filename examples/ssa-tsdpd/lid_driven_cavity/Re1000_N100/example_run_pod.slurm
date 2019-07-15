#!/bin/bash -l

#SBATCH --nodes=1 --ntasks-per-node=20

#SBATCH --job-name="N120R1000"	

#SBATCH --time=12:00:00

cd $SLURM_SUBMIT_DIR

/bin/hostname
mpirun -np $SLURM_NTASKS ../../../../src/lmp_mpi < lid_driven_cavity.lmp

