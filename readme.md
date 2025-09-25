Setup:

(if on windows this won't work, and will have to be setup manually)

- check the headers for gcc compiler location and mpi headers, this may need to be modified 
- run make install 

This should set up the virutal envrionment and install all dependancies 

- run make build 

This will build all .so files from the .pyx files in the Alg folder

To run coarse potential script, activate venv in terminal:

- source venv/bin/activate

Then run the main with a specified number of cores. To modify intial conditions, the top of the main.py file has changeable parameters (simulation time etc)

Initial condition choice: 

- using (setupData/explanatory01_pk.dat) causes Ke to explode in the case of main.py, and should only be used on main2.py and main3.py (there is no singularity avoidance with just PM method)

- mpiexec -n 4 python main.py

This should generate a file within Output. The main file will need to have the output destination changed if run more than 
once. 

analysis.ipynb provides some statistics about the simulation data. 

for main scripts: 

main 1 runs is meant to run a random configuration of particles using the coarse potential only 

main 2 runs the CLASS inital conditions using both potential types, which has a kinetic energy damping value (will nee to be changed if using main3)

main 3 runs the random simulation of particles using both potential types (will need to change damping value if using main2)

ALg contains a list of the cython files 

Analysis contains a singular python class for visualising data within the analysis.ipynb notebook

setupData contains intial conditions data. 

Output contains the generated simulation data 

Rendering is done in the rendering folder (there is an issue with the scaling of the visualisation)

There is also a copy of the report for reference 

