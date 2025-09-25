from setuptools import setup
from Cython.Build import cythonize
import numpy
import os

#This make file will only work on macOS and linux, not windows. The MPI headers were accessed via brew. 

# May need to change
os.environ["CC"] = "gcc-12"
os.environ["CXX"] = "g++-12"

# May need to change 
mpi_include_dir = "/opt/homebrew/include"
mpi_library_dir = "/opt/homebrew/lib"

cython_folder = "Alg"

extensions = cythonize(
    [
        os.path.join(cython_folder, "CLASS_init.pyx"),
        os.path.join(cython_folder, "cloud_in_a_cell.pyx"),
        os.path.join(cython_folder, "dynamics.pyx"),
        os.path.join(cython_folder, "hilbert_partition.pyx"),
        os.path.join(cython_folder, "linked_list.pyx"),
        os.path.join(cython_folder, "potential.pyx"),
        os.path.join(cython_folder, "run_coarse_sim.pyx"),
        os.path.join(cython_folder, "run_total_pot.pyx"),
    ],
    compiler_directives={"language_level": "3"},
)

for ext in extensions:
    ext.name = f"{cython_folder}.{ext.name}" 

    ext.include_dirs.append(mpi_include_dir)
    ext.library_dirs.append(mpi_library_dir)

    if ext.name in [
        f"{cython_folder}.cloud_in_a_cell",
        f"{cython_folder}.dynamics",
        f"{cython_folder}.potential",
        f"{cython_folder}.run_coarse_sim",
        f"{cython_folder}.run_total_pot",
    ]:
        ext.extra_compile_args = ["-fopenmp"]
        ext.extra_link_args = ["-fopenmp"]

setup(
    name="Sim",
    ext_modules=extensions,
    include_dirs=[numpy.get_include(), mpi_include_dir],
    zip_safe=False,
)