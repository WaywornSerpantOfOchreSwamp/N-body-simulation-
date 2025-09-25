from libc.math cimport sqrt, pow, log, ceil
import numpy as np
cimport numpy as np
from mpi4py import MPI

ctypedef np.float64_t DTYPE_t

cdef inline double compute_growth_factor(double Omega_m, double a):
    """
    Compute the growth factor approximation.
    """
    return Omega_m ** 0.55

cpdef int calculate_grid_size_power_of_two(double box_size, double target_cell_size=100.0):
    """
    Find nearest power of two for grid size 
    """
    cdef double grid_size = box_size / target_cell_size
    cdef int power_of_two = int(pow(2, ceil(log(grid_size) / log(2))))
    return power_of_two

def generate_initial_conditions(double box_size, int grid_size,double z_init, object power_spectrum_func, int rank, int size):
    """
    Generate initial conditions using the Zel'dovich approximation using a (already compiled) CLASS inital condition .dat file

    Input:
    - box_size: size of the simulation box (in Mpc/h).
    - grid_size: number of grid points along each dimension.
    - z_init: initial redshift.
    - power_spectrum_func: callable function to compute the power spectrum P(k) (change-able based on qualitly)
    - rank: MPI rank.
    - size: number of MPI processes.

    Agl:
    - Generate a space gird in fourier space, generate waevnumbers for each grid (small K means high density as k = 2pi/lambda)
    - build 3D arrays to represnt k in all three dimension
    - emulate random density fluctuations using a Gaussian random field over fourier spcae bounds
    - gather imaginary and real componets, and turn into a 2D complex array 
    - generate power specturm using scipy function (no large overhead as called once only, so not a huge difference and allows for other option)
    - apply power specturm to each dimension in fourier space 
    - use IFFT to get back to real space 
    - define particle grid according to the passed box_size and grid_sie 
    - populate particle grid with real space values, defining particle positions
    - compute velocities as a function of scale factor, scalled H param and growth factor
    - generate mass based on normalisation conditions for simulation

    Out:
    - positions: positions.
    - velocities: velocities.
    - masses: masses.
    """

    cdef double h = 0.6774  # Hubble constant in units of 100 km/s/Mpc
    cdef double Omega_m = 0.3089 # matter density parameter 
    cdef double Omega_L = 0.6911 # dark energy density parameter 
    cdef double H_0 = 100.0 * h # Hubble parameter 
    cdef double a = 1.0 / (1 + z_init) # scale factor (inverse rel to redshift)
    cdef double H = H_0 * sqrt(Omega_m * pow(a, -3) + Omega_L) # scaled hubble parameter using Friedmann eq
    cdef double f = compute_growth_factor(Omega_m, a) # growth factor due to gavitational instability (for a galaxy)

    # Fourier-space grid
    cdef np.ndarray[DTYPE_t, ndim=1] kx = np.fft.fftfreq(grid_size, d=box_size / grid_size) * 2 * np.pi
    cdef np.ndarray[DTYPE_t, ndim=1] ky = kx.copy()
    cdef np.ndarray[DTYPE_t, ndim=1] kz = kx[:grid_size // 2 + 1]
    cdef np.ndarray[DTYPE_t, ndim=3] kx_mesh, ky_mesh, kz_mesh
    kx_mesh, ky_mesh, kz_mesh = np.meshgrid(kx, ky, kz, indexing='ij', copy=True)
    cdef np.ndarray[DTYPE_t, ndim=3] k_mag = np.sqrt(kx_mesh**2 + ky_mesh**2 + kz_mesh**2)

    # Gaussian random field
    np.random.seed(rank)
    cdef Py_ssize_t shape_x = kx_mesh.shape[0]
    cdef Py_ssize_t shape_y = kx_mesh.shape[1]
    cdef Py_ssize_t shape_z = kx_mesh.shape[2]
    cdef tuple shape = (shape_x, shape_y, shape_z)

    # Generate real and imaginary parts for the random field, then 2D complex array
    cdef np.ndarray[DTYPE_t, ndim=3] real_part = np.random.normal(size=shape).astype(np.float64)
    cdef np.ndarray[DTYPE_t, ndim=3] imag_part = np.random.normal(size=shape).astype(np.float64)
    cdef np.ndarray[complex, ndim=3] random_complex = real_part + 1j * imag_part

    # Apply the power spectrum
    cdef np.ndarray[DTYPE_t, ndim=3] power_spectrum = power_spectrum_func(k_mag)
    cdef np.ndarray[complex, ndim=3] delta_k = random_complex * np.sqrt(power_spectrum / 2)

    # Compute displacement fields
    cdef np.ndarray[DTYPE_t, ndim=3] k_squared = k_mag**2
    k_squared[0, 0, 0] = 1.0  #  to avoid dividing by 0 
    cdef np.ndarray[complex, ndim=3] factor = -1j / k_squared
    cdef np.ndarray[complex, ndim=3] psi_x_k = factor * kx_mesh * delta_k
    cdef np.ndarray[complex, ndim=3] psi_y_k = factor * ky_mesh * delta_k
    cdef np.ndarray[complex, ndim=3] psi_z_k = factor * kz_mesh * delta_k

    # Apply IFFT to get back to real space 
    cdef np.ndarray[DTYPE_t, ndim=3] psi_x = np.fft.irfftn(psi_x_k, s=(grid_size, grid_size, grid_size)).real
    cdef np.ndarray[DTYPE_t, ndim=3] psi_y = np.fft.irfftn(psi_y_k, s=(grid_size, grid_size, grid_size)).real
    cdef np.ndarray[DTYPE_t, ndim=3] psi_z = np.fft.irfftn(psi_z_k, s=(grid_size, grid_size, grid_size)).real

    # Particle grid
    cdef np.ndarray[DTYPE_t, ndim=1] grid = np.linspace(0, box_size, grid_size, endpoint=False)
    cdef np.ndarray[DTYPE_t, ndim=3] grid_x, grid_y, grid_z
    grid_x, grid_y, grid_z = np.meshgrid(grid, grid, grid, indexing='ij', copy=True)
    cdef np.ndarray[DTYPE_t, ndim=1] x = (grid_x.flatten() + psi_x.flatten()) % box_size
    cdef np.ndarray[DTYPE_t, ndim=1] y = (grid_y.flatten() + psi_y.flatten()) % box_size
    cdef np.ndarray[DTYPE_t, ndim=1] z = (grid_z.flatten() + psi_z.flatten()) % box_size

    # Velocities
    cdef np.ndarray[DTYPE_t, ndim=1] vx = (a * H * f * psi_x.flatten()) / box_size
    cdef np.ndarray[DTYPE_t, ndim=1] vy = (a * H * f * psi_y.flatten()) / box_size
    cdef np.ndarray[DTYPE_t, ndim=1] vz = (a * H * f * psi_z.flatten()) / box_size

    # Masses
    cdef np.ndarray[DTYPE_t, ndim=1] masses = np.full(grid_size**3, (Omega_m * 3 * H_0**2 / (8 * np.pi * 6.67430e-11)) * (box_size / grid_size)**3)
    ## return each value individually to reduce overhead when many (10^7) particles are used
    return x, y, z, vx, vy, vz, masses