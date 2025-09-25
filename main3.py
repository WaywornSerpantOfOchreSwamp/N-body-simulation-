from mpi4py import MPI
import numpy as np 
from scipy.interpolate import interp1d
from Alg.CLASS_init import generate_initial_conditions, calculate_grid_size_power_of_two
from Alg.hilbert_partition import hilbert_partition
from Alg.run_total_pot import run_simulation 
import time 
import pandas as pd

def normalize_particles(data, box_size, total_mass):
    """
    Normalisation of particle data for random particles.  
    """
    num_particles = len(data)

    # Assign scaled particle mass
    particle_mass = total_mass / num_particles
    data['mass'] = particle_mass

    # Scale velocities using virial theorem
    velocity_scale = np.sqrt(total_mass / box_size)
    data['vx'] *= velocity_scale
    data['vy'] *= velocity_scale
    data['vz'] *= velocity_scale

    # Ensure positions are within scaled box boundaries
    data['x'] %= box_size
    data['y'] %= box_size
    data['z'] %= box_size

    return data

def main():
    """
    This calls two potential functions

    -Alg   
    - Intial conditions defined for simulation parameters
    - MPI nodes initalised 
    - Intialises system by reading from random csv configuration
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
    box_size = 1000.0 #total universe volume in Mpc/h
    grid_size = 512 #left as a power of two for greater computational efficiency, will be dynamically adjusted later 
    grid_size = calculate_grid_size_power_of_two(box_size)
    fine_cell_size = box_size / grid_size #defining cell size for fine potential calculations
    coarse_grid_size = grid_size // 2 
    coarse_cell_size = box_size / coarse_grid_size # larger gird sizes of coarse potential calcs (longer range)
    num_steps = 50000 # number of steps for simulation to run over
    dt = 0.01 # time jump between steps 
    G = 1.0
    global_density_fine = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)
    local_density = np.zeros_like(global_density_fine)
    h = fine_cell_size
    softening_length = 0.01 * h
    total_mass = 1.0 
    output_file = "Output/simulation_3.csv"
    
    start = time.time()
    if rank == 0:
        print(f"box_size: {box_size}, grid_size: {grid_size}")
        print(f"fine_cell_size: {fine_cell_size}, coarse_cell_size: {coarse_cell_size}")

    # Load particle data from CSV
    input_file = "setupData/random_particles_10e6.csv"  # Replace with the actual input CSV file path
    if rank == 0:
        particle_data = pd.read_csv(input_file)
        particle_data = normalize_particles(particle_data, box_size, total_mass)

        # Split data among ranks
        data_split = np.array_split(particle_data, size)
    else:
        data_split = None

    # Distribute data to all ranks
    local_data = comm.scatter(data_split, root=0)

    # Extract particle properties for the local rank
    x = local_data['x'].values
    y = local_data['y'].values
    z = local_data['z'].values
    vx = local_data['vx'].values
    vy = local_data['vy'].values
    vz = local_data['vz'].values
    masses = local_data['mass'].values

    x, y, z, vx, vy, vz, masses = hilbert_partition(
        x, y, z, vx, vy, vz, masses, grid_size, rank, size, fine_cell_size, box_size
    )
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