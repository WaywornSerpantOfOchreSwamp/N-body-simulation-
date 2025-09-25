cimport cython
import mpi4py.MPI as MPI 
import numpy as np
cimport numpy as np
from pyfftw.interfaces.numpy_fft import fftn, ifftn
from libc.math cimport M_PI, sqrt, floor
from cython.parallel import prange
from cpython.array cimport array

ctypedef np.float64_t DTYPE_t
cdef double PI = 3.141592653589793

cdef double cubic_spline_kernel(double r, double h) nogil:
    """
    smoothing of potenatial via cubic spline kernel
    """
    cdef double q = r / h
    cdef double sigma = 1 / (PI * h ** 3) 

    if q < 1.0:
        return sigma * (1 - 1.5 * q ** 2 + 0.75 * q ** 3)
    elif q < 2.0:
        return sigma * 0.25 * (2 - q) ** 3
    else:
        return 0.0

cpdef np.ndarray[DTYPE_t, ndim=3] compute_coarse_potential(
    np.ndarray[DTYPE_t, ndim=3] density,
    double box_size,
    double G
):
    """
    Compute the gravitational potential using FFT and slab decomposition in the z axis with MPI.

    Input:
    - density: Local slab of the density grid for each MPI rank
    - box_size: The size of the simulation box
    - G: Gravitational constant
    Alg:
     - make sure grids are correctly structured for calculation 
     - generate 1D wavevectors of gird and convert into angular wavenumbers 
     - create "slabs" by paritioning in the z axis, and hand out groups to MPI nodes (extra go to 1st), hence not using openMP 
     - use fft after transforming hte density to fourier space to get fourier density 
     - genrate 3D angular wavenumber 
     - compute poisson equation, avioding potentail 0 divison 
     - use IFFT to get back to real space 
     - copule slab potentials into one. 

    Out:
    - local slab of the gravitational potential.
    """
    # initialize MPI
    cdef int rank = MPI.COMM_WORLD.Get_rank()
    cdef int size = MPI.COMM_WORLD.Get_size()

    # define dimensions of global grid
    cdef int nx = density.shape[0]
    cdef int ny = density.shape[1]
    cdef int nz = density.shape[2]
    cdef int rfft_nz = nz // 2 + 1

    # ensure grid dimensions are powers of 2
    if not (nx & (nx - 1) == 0 and ny & (ny - 1) == 0 and nz & (nz - 1) == 0):
        raise ValueError("Grid dimensions must be powers of 2 for this implementation.")

    # calculate grid spacing
    cdef double dx = box_size / nx
    cdef double dy = box_size / ny
    cdef double dz = box_size / nz

    # global FFT frequency vectors
    cdef np.ndarray[DTYPE_t, ndim=1] kx_1d = np.fft.fftfreq(nx, d=dx) * 2 * M_PI
    cdef np.ndarray[DTYPE_t, ndim=1] ky_1d = np.fft.fftfreq(ny, d=dy) * 2 * M_PI
    cdef np.ndarray[DTYPE_t, ndim=1] kz_1d = np.fft.rfftfreq(nz, d=dz) * 2 * M_PI

    # determine the local slab for this rank
    cdef int slab_size = rfft_nz // size
    cdef int remainder = rfft_nz % size
    cdef int local_nz_start = rank * slab_size + min(rank, remainder)
    cdef int local_nz_end = local_nz_start + slab_size
    if rank < remainder:
        local_nz_end += 1

    # ensure valid slice range
    if local_nz_start >= rfft_nz or local_nz_end > rfft_nz:
        raise ValueError(f"Rank {rank} has invalid slab range: start={local_nz_start}, end={local_nz_end}")

    # slice density for this rank
    cdef np.ndarray[DTYPE_t, ndim=3] local_density = density[:, :, local_nz_start:local_nz_end]

    # perform FFT on the local density
    cdef np.ndarray[complex, ndim=3] local_density_k = fftn(local_density, axes=(0, 1, 2))

    # slice kz for this rank
    cdef np.ndarray[DTYPE_t, ndim=1] kz = kz_1d[local_nz_start:local_nz_end]

    # create 3D meshgrid for the local slab
    cdef np.ndarray[DTYPE_t, ndim=3] kx, ky, kz_mesh
    kx, ky, kz_mesh = np.meshgrid(kx_1d, ky_1d, kz, indexing='ij')
    cdef np.ndarray[DTYPE_t, ndim=3] k_squared = kx**2 + ky**2 + kz_mesh**2
    k_squared[k_squared == 0] = 1  # avoid division by zero

    # compute potential in Fourier space
    cdef np.ndarray[complex, ndim=3] local_potential_k = -4 * M_PI * G * local_density_k / k_squared
    if rank == 0:
        local_potential_k[0, 0, 0] = 0  # remove mean potential

    # transform back to real space
    cdef np.ndarray[DTYPE_t, ndim=3] local_potential = ifftn(local_potential_k, axes=(0, 1, 2)).real

    # extract shape dimensions and convert to a tuple
    cdef int dim_x = local_potential.shape[0]
    cdef int dim_y = local_potential.shape[1]
    cdef int dim_z = local_potential.shape[2]

    # debug potential shape
    print(f"Rank {rank}: local_potential.shape={(dim_x, dim_y, dim_z)}")

    return local_potential

cpdef np.ndarray[np.float64_t, ndim=3] compute_fine_potential(
    double[:] x, double[:] y, double[:] z, double[:] mass,
    np.ndarray[np.int32_t, ndim=1] particle_cell_map,
    np.ndarray[np.int32_t, ndim=1] cell_start,
    double h, double softening_length, double cell_size,
    int grid_size, double box_size, int cell_start_len
):
    """
    Compute the potential on a 3D grid using particle data and linked list mapping. Pair interaction only 

    Input: 
    - x,y,z mass : particle parameters 
    - particle_cell_map : mapping of particles to grid spaces 
    - cell_start : starting index of cells
    - h: smoothing scale in cubic_spline_kernel 
    - softening_length: reduces impact of sinularities 
    - cell_size: cell size
    - grid_szie: grid size 
    - boz_size : box size
    - cell_start_len : legnth of cell size

    Alg
    - computes pairwise potential between adjacent particles 
    """
    cdef np.ndarray[np.float64_t, ndim=3] potential = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)
    cdef int i, j, ix, iy, iz, cell_index, neighbor_index
    cdef int dx, dy, dz  
    cdef double r, w, dx_part, dy_part, dz_part
    cdef int dx_vals[3]

    dx_vals[0] = -1
    dx_vals[1] = 0
    dx_vals[2] = 1

    # iterate over all particles to compute contributions to the potential grid
    for i in range(x.shape[0]):
        ix = <int>(floor(x[i] / cell_size)) % grid_size
        iy = <int>(floor(y[i] / cell_size)) % grid_size
        iz = <int>(floor(z[i] / cell_size)) % grid_size
        cell_index = ix * grid_size * grid_size + iy * grid_size + iz

        for dx in dx_vals:
            for dy in dx_vals:
                for dz in dx_vals:
                    neighbor_index = (
                        ((ix + dx + grid_size) % grid_size) * grid_size * grid_size +
                        ((iy + dy + grid_size) % grid_size) * grid_size +
                        ((iz + dz + grid_size) % grid_size)
                    )

                    if neighbor_index < 0 or neighbor_index >= cell_start_len - 1:
                        continue

                    # iterate over particles in neighboring cells
                    for j in range(cell_start[neighbor_index], cell_start[neighbor_index + 1]):
                        dx_part = x[i] - x[particle_cell_map[j]]
                        dy_part = y[i] - y[particle_cell_map[j]]
                        dz_part = z[i] - z[particle_cell_map[j]]
                        r = sqrt(dx_part ** 2 + dy_part ** 2 + dz_part ** 2 + softening_length ** 2)

                        # add contribution to the grid-based potential
                        potential[ix, iy, iz] -= mass[particle_cell_map[j]] / r

    return potential