OBJ_DIR = ./

UNAME=$(shell uname)

SRCS =  fsi.lmp
	
NP = 4
BIN = ../../../src/lmp_mpi
RUNCMD = mpirun -np $(NP) $(BIN) -in $(SRCS)

run :	
	clear
	$(RUNCMD)

clean :
	rm -rf *.vtk
	rm -rf log.*
	rm -rf *.lammpstrj
	rm -rf *.profile
	rm -rf *.dat
