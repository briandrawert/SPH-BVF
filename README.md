# SPH-BVF
### A smoothed particle hydrodynamics transport-velocity formulation for fluid-structure interaction with fixed and moving boundaries

by:  Bruno Jacob, Brian Drawert, Tau-Mu Yi, Linda Petzold


We present a new weakly-compressible smoothed particle hydrodynamics (SPH) method capable of modeling non-slip fixed and moving wall boundary conditions. The formulation combines a boundary volume fraction (BVF) wall approach with the transport-velocity SPH method. The resulting method, named SPH-BVF, offers detection of arbitrary shaped solid walls on-the-fly with small computational overhead due to its local formulation. This simple framework is capable of solving problems that are difficult or infeasible for standard SPH, namely flows subject to large shear stresses or at moderate Reynolds numbers, and mass transfer in deformable boundaries. In addition, the method extends the transport-velocity formulation to reaction- diffusion transport of mass in Newtonian fluids and linear elastic solids, which is common in biological structures. Taken together, the SPH-BVF method provides a good balance of simplicity and versatility, while avoiding some of the standard obstacles associated with SPH: particle penetration at the boundaries, tension instabilities and anisotropic particle alignments, that hamper SPH from being applied to complex problems, such as fluid-structure interaction in a biological system.

Keywords: Solid wall model, Transport-velocity, Smoothed particle hydrodynamics, Boundary condition, Deforming boundaries



To run the examples found in the paper, you can use our pre-configued docker containers.  First make sure you have [Docker installed](https://www.docker.com/) and running.  Then type the following commands into the terminal to run each of the examples.  The result will be to create a number of VTK files in the folders within the current directory.  You can view these files with a program such as [ParaView](https://www.paraview.org/).

From the main text: 

### lid-driven cavity flow
```
docker run -it -v "`pwd`":/work  briandrawert/sph_bvf /bin/bash -c "/run_lid_driven_cavity_flow.sh"
```

### natural convection
```
docker run -it -v "`pwd`":/work  briandrawert/sph_bvf /bin/bash -c "/run_natural_convection.sh"
```

### fluid-structure interaction
```
docker run -it -v "`pwd`":/work  briandrawert/sph_bvf /bin/bash -c "/run_fsi.sh"
```

### yeast cell mating projection growth
```
docker run -it -v "`pwd`":/work  briandrawert/sph_bvf /bin/bash -c "/run_cell_polarization.sh"
```


