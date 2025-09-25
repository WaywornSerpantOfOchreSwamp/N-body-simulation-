import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
import pandas as pd

class SimulationAnalysis:
    def __init__(self, data):
        """
        initialise the SimulationAnalyzer with simulation data.
        """
        self.data = data
    
    def compute_position_statistics(self):
        """
        compute the variance in position for the simulation.
        """
        stats = self.data[['x', 'y', 'z']].agg(['mean', 'std', 'var', 'min', 'max'])
        print("\nPosition Statistics (X, Y, Z):")
        print(stats)
        return stats
    
    def check_variance_by_timestep(self):
        """
        visualize the variance of position of each particle over each timestep.
        """
        variance_data = self.data.groupby('timestep')[['x', 'y', 'z']].var()
        print("\nVariance Across Timesteps:")
        print(variance_data)
        
        plt.figure()
        plt.plot(variance_data.index, variance_data['x'], label='Variance in X')
        plt.plot(variance_data.index, variance_data['y'], label='Variance in Y')
        plt.plot(variance_data.index, variance_data['z'], label='Variance in Z')
        plt.title("Variance of Positions Over Timesteps")
        plt.xlabel("Timestep")
        plt.ylabel("Variance")
        plt.legend()
        plt.show()
    
    def visualise_trajectories(self, particle_ids, timesteps):
        """
        for particular particles, plot their trajectory in space over a certain timestep.
        """
        for pid in particle_ids:
            subset = self.data[self.data['particle_id'] == pid].head(timesteps)
            print(f"Trajectory of particle {pid}:")
            print(subset[['timestep', 'x', 'y', 'z']].to_string(index=False))
            
            plt.plot(subset['timestep'], subset['x'], label=f'Particle {pid} (X)')
            plt.plot(subset['timestep'], subset['y'], label=f'Particle {pid} (Y)')
            plt.plot(subset['timestep'], subset['z'], label=f'Particle {pid} (Z)')
        plt.title("Particle Trajectories Over Time")
        plt.xlabel("Timestep")
        plt.ylabel("Position")
        plt.legend()
        plt.show()
    
    def compute_kinetic_energy(self):
        """
        Compute and visualize the kinetic energy over all timesteps.
        """
        self.data['kinetic_energy'] = 0.5 * self.data['mass'] * (
            self.data['vx']**2 + self.data['vy']**2 + self.data['vz']**2
        )
        
        energy_by_timestep = self.data.groupby('timestep')['kinetic_energy'].sum()
        print("\nTotal Kinetic Energy at Each Timestep:")
        print(energy_by_timestep)
        
        plt.figure()
        plt.plot(energy_by_timestep.index, energy_by_timestep.values, label="Total Kinetic Energy")
        plt.title("Total Kinetic Energy Over Time")
        plt.xlabel("Timestep")
        plt.ylabel("Total Kinetic Energy")
        plt.legend()
        plt.show()
        
        return energy_by_timestep
    
    def visualise_single_particle(self, particle_id):
        """
        Visualize a specific particle's trajectory.
        """
        particle_data = self.data[self.data['particle_id'] == particle_id]
        
        if particle_data.empty:
            print(f"No data found for particle ID {particle_id}.")
            return
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(particle_data['x'], particle_data['y'], particle_data['z'],
                        c=particle_data['timestep'], cmap='viridis', s=20)
        plt.colorbar(sc, label="Timestep")
        ax.set_title(f"Trajectory of Particle ID {particle_id}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()
    
    def visualie_velocity_distribution(self, timestep):
        """
        Plot the velocity distribution for a given timestep.
        """
        # Access self.data to refer to the data attribute of the class
        timestep_data = self.data[self.data['timestep'] == timestep]
        velocities = np.sqrt(
            timestep_data['vx']**2 + timestep_data['vy']**2 + timestep_data['vz']**2
        )

        plt.figure()
        plt.hist(velocities, bins=50, density=True, alpha=0.7, label="Velocity Distribution")
        plt.title(f"Velocity Distribution at Timestep {timestep}")
        plt.xlabel("Velocity")
        plt.ylabel("Probability Density")
        plt.legend()
        plt.show()