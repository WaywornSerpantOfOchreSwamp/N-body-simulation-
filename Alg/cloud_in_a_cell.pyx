from mpi4py import MPI
import numpy as np
from cython.parallel import prange
from libc.math cimport fmin  

cpdef cloud_in_cell_mpi_openmp(double[:, :, :] mesh, double[:] x, double[:] y, double[:] z, double[:] mass, double cell_size):
    """
    Assign particle mass to a 3D mesh using MPI and OpenMP.

    Input: 
    - mesh(double[:, :, :]): 3D numpy array represeting the total mesh (or complete gird space)
    - x, y, z (double[:]): paricle positions in cartiesian coordinates 
    - mass(double[:]): mass of each particle 
    - cell_szie(double): unit cell of mesh, smallest space of mass density 

    Alg: 
    - get mpi rank and number of process 
    - initliase the mesh, then generate a local mesh for each MPI component 
    - partition particles by total number based on how MPI ranks there are
    - assign each process a subset of these particles
    - for each particle in each MPI rank, use openMP to compute the cloud in the cell function 
        - Find the cell indicies containing the particle
        - calculate the fractional distance from the center of the cell to the particle 
        - distribute the mass across the 8 nearest cells 
    - update the local mesh 
    - go from each individual MPI thread back to a main thread on the root process 
    - return the global mesh

    Out:
    - Mass density of each grid

    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    cdef int nx = mesh.shape[0]
    cdef int ny = mesh.shape[1]
    cdef int nz = mesh.shape[2]

    cdef double[:, :, :] local_mesh = np.zeros((nx, ny, nz), dtype=np.float64)

    cdef int num_particles = len(x)
    cdef int particles_per_process = num_particles // size
    cdef int start = rank * particles_per_process
    cdef int end = (rank + 1) * particles_per_process if rank != size - 1 else num_particles

    cdef int i, j, k, l, ix, iy, iz, cell_x, cell_y, cell_z
    cdef double dx, dy, dz, fx, fy, fz, weight

    for i in prange(start, end, nogil=True): 
        ix = int(x[i] // cell_size)
        iy = int(y[i] // cell_size)
        iz = int(z[i] // cell_size)

        dx = (x[i] % cell_size) / cell_size
        dy = (y[i] % cell_size) / cell_size
        dz = (z[i] % cell_size) / cell_size

        for j in range(2):
            for k in range(2):
                for l in range(2):
                    fx = (1 - dx) if j == 0 else dx
                    fy = (1 - dy) if k == 0 else dy
                    fz = (1 - dz) if l == 0 else dz
                    weight = fx * fy * fz

                    cell_x = ix + j if ix + j < nx else nx - 1
                    cell_y = iy + k if iy + k < ny else ny - 1
                    cell_z = iz + l if iz + l < nz else nz - 1

                    local_mesh[cell_x, cell_y, cell_z] += mass[i] * weight
    comm.Reduce(local_mesh, mesh, op=MPI.SUM, root=0)