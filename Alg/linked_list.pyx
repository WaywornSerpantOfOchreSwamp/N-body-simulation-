from libc.math cimport floor
import numpy as np
cimport numpy as np

cpdef tuple construct_linked_list(
    np.ndarray[np.float64_t, ndim=1] x,
    np.ndarray[np.float64_t, ndim=1] y,
    np.ndarray[np.float64_t, ndim=1] z,
    double cell_size,
    int grid_size
):
    """
    Construct a linked list for particle-cell mapping.
    Out:
    - Cell-to-particle mapping (flattened array).
    - Starting index for each cell in the flattened array.
    """
    cdef int i, ix, iy, iz, cell_index
    cdef int num_cells = grid_size ** 3
    cdef int[:] cell_count = np.zeros(num_cells, dtype=np.int32)
    cdef int[:] cell_start = np.zeros(num_cells, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] particle_cell_map = np.zeros(x.shape[0], dtype=np.int32)

    # Calculate which cell each particle belongs to
    for i in range(x.shape[0]):
        ix = <int>(floor(x[i] / cell_size)) % grid_size
        iy = <int>(floor(y[i] / cell_size)) % grid_size
        iz = <int>(floor(z[i] / cell_size)) % grid_size
        cell_index = ix * grid_size * grid_size + iy * grid_size + iz
        cell_count[cell_index] += 1

    cdef int cumulative_count = 0
    for i in range(num_cells):
        cell_start[i] = cumulative_count
        cumulative_count += cell_count[i]

    # fill particle-cell mapping
    cell_count[:] = 0  
    for i in range(x.shape[0]):
        ix = <int>(floor(x[i] / cell_size)) % grid_size
        iy = <int>(floor(y[i] / cell_size)) % grid_size
        iz = <int>(floor(z[i] / cell_size)) % grid_size
        cell_index = ix * grid_size * grid_size + iy * grid_size + iz
        particle_cell_map[cell_start[cell_index] + cell_count[cell_index]] = i
        cell_count[cell_index] += 1

    return particle_cell_map, np.asarray(cell_start)