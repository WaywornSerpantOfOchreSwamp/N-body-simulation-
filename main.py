from mpi4py import MPI
import numpy as np
from scipy.interpolate import interp1d
from Alg.CLASS_init import calculate_grid_size_power_of_two
from Alg.hilbert_partition import hilbert_partition
from Alg.run_coarse_sim import run_simulation
import time
import pandas as pd

def normalize_particles(data, box_size, total_mass):
    """
    Normalize particle positions, velocities, and masses for a scaled simulation.

    Parameters:
        data (pd.DataFrame): Particle data.
        box_size (float): Box size (scaled).
        total_mass (float): Total mass in the box (scaled).

    Returns:
        pd.DataFrame: Normalized particle data.
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
    This calls only a singular potential value

    -Alg   
    - Intial conditions defined for simulation parameters
    - MPI nodes initalised 
    - Intialises system by reading from random csv configuration
    - apply initial hilbert indicies to pass to run script 
    - run main simulation loop

    Out: 
    - csv of particle position for each timestep for rendering
    """
    # Setup MPI for initializing grid and timestep calculations
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Scaled universe parameters
    box_size = 1000.0  
    grid_size = 128  # Must remain a power of 2
    fine_cell_size = box_size / grid_size
    coarse_cell_size = box_size / (grid_size // 2)
    num_steps = 1000
    dt = 0.01  # Normalized time step
    G = 1.0  # Normalized gravitational constant
    total_mass = 1.0  # Normalized total mass
    output_file = "Output/simulation_1.csv"

    # Will catch error is cell size is less than 1
    assert fine_cell_size > 1, f"fine_cell_size is too small: {fine_cell_size}"
    assert coarse_cell_size > 1, f"coarse_cell_size is too small: {coarse_cell_size}"

    # Load particle data from CSV
    input_file = "setupData/particles_1000.csv"  # can be modified
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

    # assign a Hilbert (local) index to each particle for locality in MPI threads
    x, y, z, vx, vy, vz, masses = hilbert_partition(
        x, y, z, vx, vy, vz, masses, grid_size, rank, size, fine_cell_size, box_size
    )
    x_prev = np.copy(x)
    y_prev = np.copy(y)
    z_prev = np.copy(z)

    # Pass to the simulation
    run_simulation(
        num_steps=num_steps,
        global_density_fine=np.zeros((grid_size, grid_size, grid_size), dtype=np.float64),
        local_density=np.zeros((grid_size, grid_size, grid_size), dtype=np.float64),
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
    )

if __name__ == "__main__":
    main()
    