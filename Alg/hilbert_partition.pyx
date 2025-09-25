import numpy as np
cimport numpy as np
from libc.math cimport floor, log2, ceil, fmod

def calculate_hilbert_indices(x, y, z, grid_size, cell_size, p):
    """
    Compute Hilbert indices.
    Input:
    - x, y, z: particle positions.
    - grid_size: grid size along each dimension.
    - cell_size: size of each grid cell.
    - p: order of the Hilbert curve.

    Alg:
    - import hilbertcurve function
    - for each particle, calculate index 

    Out:
    - array of Hilbert indices.
    """
    from hilbertcurve.hilbertcurve import HilbertCurve
    hilbert_curve = HilbertCurve(p, 3)
    indices = []
    for i in range(len(x)):
        ix = min(max(int(x[i] // cell_size), 0), grid_size - 1)
        iy = min(max(int(y[i] // cell_size), 0), grid_size - 1)
        iz = min(max(int(z[i] // cell_size), 0), grid_size - 1)
        indices.append(hilbert_curve.distance_from_point([ix, iy, iz]))
    return np.array(indices, dtype=np.uint64)

cpdef tuple hilbert_partition(
    double[:] x, double[:] y, double[:] z,
    double[:] vx, double[:] vy, double[:] vz,
    double[:] mass,
    int grid_size, int rank, int size, double cell_size, double box_size
):
    """
    Partition particles based on a 3D Hilbert curve.

    Input:
    - x, y, z: arrays of particle positions.
    - vx, vy, vx: velocities to also tag 
    - mass: mass to tag
    - grid_size: number of cells along each dimension of the grid.
    - rank: MPI rank.
    - size: number of MPI ranks.
    - cell_size: size of each grid cell.

    Alg:
    - ensure particles are within the box 
    - calculate hilbert indicies 
    - sort indicies 
    - assign particles to mpi rank 

    Out:
    - local partition of particle indices for the current rank.
    """
    cdef int p = <int>ceil(log2(grid_size))  
    cdef np.ndarray[np.uint64_t, ndim=1] indices
    cdef np.ndarray[np.int64_t, ndim=1] sorted_indices
    cdef int num_particles = x.shape[0]
    cdef int start, end, extra
    cdef int i

    cdef np.ndarray[np.float64_t, ndim=1] x_np = np.asarray(x)
    cdef np.ndarray[np.float64_t, ndim=1] y_np = np.asarray(y)
    cdef np.ndarray[np.float64_t, ndim=1] z_np = np.asarray(z)
    cdef np.ndarray[np.float64_t, ndim=1] vx_np = np.asarray(vx)
    cdef np.ndarray[np.float64_t, ndim=1] vy_np = np.asarray(vy)
    cdef np.ndarray[np.float64_t, ndim=1] vz_np = np.asarray(vz)
    cdef np.ndarray[np.float64_t, ndim=1] mass_np = np.asarray(mass)

    # wrap particles inside of box
    for i in range(num_particles):
        x_np[i] = fmod(x_np[i], box_size)
        y_np[i] = fmod(y_np[i], box_size)
        z_np[i] = fmod(z_np[i], box_size)

    indices = calculate_hilbert_indices(x_np, y_np, z_np, grid_size, cell_size, p)
    sorted_indices = np.argsort(indices)

    # assing particles to MPI rank
    start = rank * (num_particles // size)
    end = (rank + 1) * (num_particles // size)
    extra = num_particles % size

    if rank < extra:
        start += rank
        end += rank + 1
    else:
        start += extra
        end += extra
    
    cdef np.ndarray[np.int64_t, ndim=1] local_indices = sorted_indices[start:end].astype(np.int64)

    return (
        x_np[local_indices],
        y_np[local_indices],
        z_np[local_indices],
        vx_np[local_indices],
        vy_np[local_indices],
        vz_np[local_indices],
        mass_np[local_indices],
    )
