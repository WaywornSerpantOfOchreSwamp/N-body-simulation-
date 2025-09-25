from libc.math cimport sqrt 
import numpy as np 
cimport numpy as np 

cdef inline void compute_acceleration(
    np.ndarray[np.float64_t, ndim=3] potential, 
    double[:] x, double[:] y, double[:] z, 
    double[:] ax, double[:] ay, double[:] az, 
    double cell_size, 
    int local_nz_start, int local_nz_end
):
    """
    Calculate acceleration based on potential. For uneven splliting of potential slabs on the z-axis, this 
    code may lead to a single MPI node recieving a width of 1, which results in acceleration in and out of the cell without 
    introducing stable boundary conditons.  
    """
    cdef int i, ix, iy, iz
    cdef double fx, fy, fz
    cdef int nx = potential.shape[0]
    cdef int ny = potential.shape[1]
    cdef int nz = potential.shape[2]

    for i in range(x.shape[0]):
        ix = int(x[i] // cell_size) % nx
        iy = int(y[i] // cell_size) % ny
        iz = int(z[i] // cell_size) % nz

        # compute fx 
        if ix == 0:
            fx = (potential[ix + 1, iy, iz] - potential[ix, iy, iz]) / cell_size
        elif ix == nx - 1:
            fx = (potential[ix, iy, iz] - potential[ix - 1, iy, iz]) / cell_size
        else:
            fx = (potential[ix + 1, iy, iz] - potential[ix - 1, iy, iz]) / (2 * cell_size)

        # compute fy
        if iy == 0:
            fy = (potential[ix, iy + 1, iz] - potential[ix, iy, iz]) / cell_size
        elif iy == ny - 1:
            fy = (potential[ix, iy, iz] - potential[ix, iy - 1, iz]) / cell_size
        else:
            fy = (potential[ix, iy + 1, iz] - potential[ix, iy - 1, iz]) / (2 * cell_size)

        # compute fz with slab-specific handling
        if nz == 1:  # if there is single slice in z-direction, there is no gradient
            fz = 0.0 
        elif iz == 0:  # First slice in slab
            fz = (potential[ix, iy, iz + 1] - potential[ix, iy, iz]) / cell_size
        elif iz == nz - 1:  # Last slice in slab
            fz = (potential[ix, iy, iz] - potential[ix, iy, iz - 1]) / cell_size
        else:  # central difference for internal slices
            fz = (potential[ix, iy, iz + 1] - potential[ix, iy, iz - 1]) / (2 * cell_size)

        ax[i] = -fx
        ay[i] = -fy
        az[i] = -fz

cpdef update_particles_verlet(
    double[:] x, double[:] y, double[:] z,
    double[:] x_prev, double[:] y_prev, double[:] z_prev,
    double[:] ax, double[:] ay, double[:] az,
    np.ndarray[np.float64_t, ndim=3] potential,
    double dt, double box_size, double cell_size,
    int local_nz_start, int local_nz_end
):
    """
    Update particle positions using Verlet integration.

    Input:
    - x, y, z: current positions.
    - x_prev, y_prev, z_prev: previous positions.
    - ax, ay, az: accelerations
    - potential: 3D potential grid.
    - dt: time step.
    - box_size: size of the simulation box (for periodic boundaries).
    - cell_size: size of grid cell.
    """
    cdef int i, n_particles = x.shape[0]
    cdef double x_new, y_new, z_new

    compute_acceleration(potential, x, y, z, ax, ay, az, cell_size, local_nz_start, local_nz_end)

    for i in range(n_particles):
        x_new = 2 * x[i] - x_prev[i] + ax[i] * dt**2
        y_new = 2 * y[i] - y_prev[i] + ay[i] * dt**2
        z_new = 2 * z[i] - z_prev[i] + az[i] * dt**2

        x_new %= box_size
        y_new %= box_size
        z_new %= box_size

        x_prev[i] = x[i]
        y_prev[i] = y[i]
        z_prev[i] = z[i]

        x[i] = x_new
        y[i] = y_new
        z[i] = z_new

cpdef void rescale_velocities(
    double[:] vx, double[:] vy, double[:] vz, 
    double[:] mass, double ke_target
):
    """
    Rescale velocities to keep kinetic energy under a threshold.
    
    Input:
    - vx, vy, vz: Velocity components.
    - mass: Particle masses.
    - ke_target: Target kinetic energy.
    """
    cdef int i, n_particles = vx.shape[0]
    cdef double ke = 0.0, v_squared, scaling_factor
    
    # calculate kinetic energy
    for i in range(n_particles):
        v_squared = vx[i]**2 + vy[i]**2 + vz[i]**2
        ke += 0.5 * mass[i] * v_squared

    # enfroce ke limit by refactoring all velocites over this limit
    if ke > ke_target:
        scaling_factor = (ke_target / ke) ** 0.5
        for i in range(n_particles):
            vx[i] *= scaling_factor
            vy[i] *= scaling_factor
            vz[i] *= scaling_factor

cpdef double compute_total_kinetic_energy(
    np.ndarray[np.float64_t, ndim=1] vx,
    np.ndarray[np.float64_t, ndim=1] vy,
    np.ndarray[np.float64_t, ndim=1] vz,
    np.ndarray[np.float64_t, ndim=1] mass
):
    cdef int i, n_particles = vx.shape[0]
    cdef double ke = 0.0

    for i in range(n_particles):
        ke += 0.5 * mass[i] * (vx[i]**2 + vy[i]**2 + vz[i]**2)

    return ke