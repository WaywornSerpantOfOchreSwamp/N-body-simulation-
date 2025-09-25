from libc.math cimport floor
import numpy as np
cimport numpy as np
from mpi4py.MPI cimport Comm 
from mpi4py import MPI
from Alg.cloud_in_a_cell import cloud_in_cell_mpi_openmp
from Alg.potential import compute_coarse_potential
from Alg.dynamics import update_particles_verlet
from Alg.linked_list import construct_linked_list
from Alg.hilbert_partition import hilbert_partition
from time import time 

def run_simulation(
    int num_steps,
    np.ndarray[np.float64_t, ndim=3] global_density_fine,
    np.ndarray[np.float64_t, ndim=3] local_density,
    np.ndarray[np.float64_t, ndim=1] x,
    np.ndarray[np.float64_t, ndim=1] y,
    np.ndarray[np.float64_t, ndim=1] z,
    np.ndarray[np.float64_t, ndim=1] x_prev,  
    np.ndarray[np.float64_t, ndim=1] y_prev,  
    np.ndarray[np.float64_t, ndim=1] z_prev,  
    np.ndarray[np.float64_t, ndim=1] vx,
    np.ndarray[np.float64_t, ndim=1] vy,
    np.ndarray[np.float64_t, ndim=1] vz,
    np.ndarray[np.float64_t, ndim=1] mass,
    double fine_cell_size,
    double box_size,
    double G,
    double dt,
    int coarse_cell_size,
    Comm comm,
    int rank,
    int size,
    str output_file,
    double cell_size,   
    int grid_size,
):  
    """
    Simulation run script for singular potential: 

    Alg
    - initalise parameters neccessary for computation 
    - apply cloud in a cell across MPI ranks 
    - combine values together 
    - compute coarse potential (PM main method)
    - apply dynamics 
    - if step is a multiple of 10, write system values to csv
    - if step is a multiple of 10, re-assign hilbert index
    
    """
    cdef int step, num_particles
    cdef list all_data = []
    cdef np.ndarray[np.float64_t, ndim=3] coarse_potential
    cdef np.ndarray[np.float64_t, ndim=1] combined_x, combined_y, combined_z, combined_vx, combined_vy, combined_vz, combined_mass
    cdef np.ndarray[np.float64_t, ndim=1] ax, ay, az

    num_particles = x.shape[0]
    particle_cell_map, cell_start = construct_linked_list(x, y, z, cell_size, grid_size)

    # initialize accelerations
    ax = np.zeros_like(x)
    ay = np.zeros_like(y)
    az = np.zeros_like(z)

    # calculate slab boundaries for this rank
    rfft_nz = grid_size // 2 + 1
    slab_size = rfft_nz // size
    remainder = rfft_nz % size
    local_nz_start = rank * slab_size + min(rank, remainder)
    local_nz_end = local_nz_start + slab_size
    if rank < remainder:
        local_nz_end += 1

    for step in range(num_steps):
        print(f"[Rank {rank}] Step {step} started.")
        start_step = time()

        if step % 10 == 0:
            x, y, z, vx, vy, vz, mass = hilbert_partition(
                x, y, z, vx, vy, vz, mass,
                grid_size, rank, size, fine_cell_size, box_size
            )
            particle_cell_map, cell_start = construct_linked_list(x, y, z, cell_size, grid_size)
        global_density_fine.fill(0.0)
        local_density.fill(0.0)

        cloud_in_cell_mpi_openmp(global_density_fine, x, y, z, mass, fine_cell_size)
        MPI.Comm.Allreduce(comm, MPI.IN_PLACE, global_density_fine, MPI.SUM)
        coarse_potential = compute_coarse_potential(global_density_fine, box_size, G)

        if step == 0:
            x_prev = x - vx * dt
            y_prev = y - vy * dt
            z_prev = z - vz * dt
            x_prev %= box_size
            y_prev %= box_size
            z_prev %= box_size

        # Update particle positions
        update_particles_verlet(
            x, y, z, x_prev, y_prev, z_prev,
            ax, ay, az, coarse_potential,
            dt, box_size, coarse_cell_size,
            local_nz_start, local_nz_end
        )

        # Gather data for output
        all_x = comm.gather(x, root=0)
        all_y = comm.gather(y, root=0)
        all_z = comm.gather(z, root=0)
        all_vx = comm.gather(vx, root=0)
        all_vy = comm.gather(vy, root=0)
        all_vz = comm.gather(vz, root=0)
        all_mass = comm.gather(mass, root=0)

        if step % 10 == 0 and rank == 0:
            if all_x and all_y and all_z:
                combined_x = np.concatenate(all_x)
                combined_y = np.concatenate(all_y)
                combined_z = np.concatenate(all_z)
                combined_vx = np.concatenate(all_vx)
                combined_vy = np.concatenate(all_vy)
                combined_vz = np.concatenate(all_vz)
                combined_mass = np.concatenate(all_mass)

                all_data.append((step, combined_x, combined_y, combined_z, combined_vx, combined_vy, combined_vz, combined_mass))

    # Write output to CSV
    if rank == 0:
        with open(output_file, "w") as csvfile:
            csvfile.write("timestep,particle_id,x,y,z,vx,vy,vz,mass\n")
            for new_step, (step, cx, cy, cz, cvx, cvy, cvz, cmass) in enumerate(all_data, start=1):
                for i in range(cx.shape[0]):
                    csvfile.write(f"{new_step},{i},{cx[i]},{cy[i]},{cz[i]},{cvx[i]},{cvy[i]},{cvz[i]},{cmass[i]}\n")

