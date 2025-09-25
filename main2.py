from mpi4py import MPI
import numpy as np 
from scipy.interpolate import interp1d
from Alg.CLASS_init import generate_initial_conditions, calculate_grid_size_power_of_two
from Alg.hilbert_partition import hilbert_partition
from Alg.run_total_pot import run_simulation 
import time 

def main():
    """
    This calls two potential functions 

    -Alg   
    - Intial conditions defined for simulation parameters
    - MPI nodes initalised 
    - Intialises system by reading in power spectrum from CLASS dataset
    - apply initial hilbert indicies to pass to run script 
    - run main simulation loop

    Out: 
    - csv of particle position for each timestep for rendering
    """

    #Setup MPI for Initalising grid and for timestep calcs 

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Initalise finite universe parameters (or universe gird)
    box_size = 20000 #total universe volume in Mpc/h
    grid_size = 16384 #left as a power of two for greater computational efficiency, will be dynamically adjusted later 
    grid_size = calculate_grid_size_power_of_two(box_size)
    fine_cell_size = box_size / grid_size #defining cell size for fine potential calculations
    coarse_grid_size = grid_size // 2 
    coarse_cell_size = box_size / coarse_grid_size # larger gird sizes of coarse potential calcs (longer range)
    z_init = 0 #inital red-shift parameter 
    power_spectrum_file = "setupData/explanatory01_pk.dat" ## generated via the CLASS system 
    num_steps = 500 # number of steps for simulation to run over
    dt = 0.01 # time jump between steps 
    G = 6.67430e-11 / box_size**3 # normalised G value
    global_density_fine = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)
    local_density = np.zeros_like(global_density_fine)
    h = fine_cell_size
    softening_length = 0.01 * h
    output_file = "Output/simulation_2.csv"
    
    start = time.time()
    # initalise power spec file onto master thread and broadcast paritions to others
    if rank == 0: 
        data = np.loadtxt(power_spectrum_file)
        k_values = data[:, 0] #wavenumber
        Pk_values = data[:, 1] # power specturm 
    else:
        if rank != 0: 
            k_values, Pk_values = None, None ##handle error here
    
    ## pass power spectrum data to all availble ranks 
    k_values = comm.bcast(k_values, root=0)
    Pk_values = comm.bcast(Pk_values, root=0)

    #Generate power specturm ready for intialisation handelling 
    power_spectrum_func = interp1d(k_values, Pk_values, kind='cubic', bounds_error = False, fill_value=0.0)

    #Initalise galaxy using Zel'dovich approximation
    x, y, z, vx, vy, vz, masses = generate_initial_conditions(box_size, grid_size, z_init, power_spectrum_func, rank, size)
    
    # assign a hilbert (local) index to each particle to insure MPI threads contain local particles for quicker calculations 
    x, y, z, vx, vy, vz, masses = hilbert_partition(x, y, z, vx, vy, vz, masses, grid_size, rank, size, fine_cell_size, box_size)
    x_prev = np.copy(x)
    y_prev = np.copy(y)
    z_prev = np.copy(z)

    run_simulation(num_steps=num_steps,
    global_density_fine=global_density_fine,
    local_density=local_density,
    x=x,
    y=y,
    z=z,
    x_prev=x_prev,
    y_prev=y_prev,
    z_prev=z_prev,
    vx=vx,
    vy=vy,
    vz=vz,
    mass=masses,
    fine_cell_size=fine_cell_size,
    box_size=box_size,
    G=G,
    dt=dt,
    coarse_cell_size=coarse_cell_size,
    comm=comm,
    rank=rank,
    size=size,
    output_file=output_file,
    cell_size=fine_cell_size,
    grid_size=grid_size,
    h = h, 
    softening_length = softening_length)

    end = time.time()
    print(f"Time for computation: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()