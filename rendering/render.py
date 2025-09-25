from manim import ThreeDScene, ThreeDAxes, VGroup, Dot3D, FadeIn, FadeOut
import numpy as np
import pandas as pd

class NBodySimulation(ThreeDScene):
    def construct(self):
        """
        This class renders a scene using the libary manin, which will visualise simulations. This can only be run properly with a small amount of particles. 
        """
        data = pd.read_csv("Output/simulation_scaled.csv") # enter target file here

        x_min, x_max = data['x'].min(), data['x'].max()
        y_min, y_max = data['y'].min(), data['y'].max()
        z_min, z_max = data['z'].min(), data['z'].max()

        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        scale_factor = max_range  
        data['x'] = (data['x'] - x_min) / scale_factor  
        data['y'] = (data['y'] - y_min) / scale_factor
        data['z'] = (data['z'] - z_min) / scale_factor

        timesteps = np.unique(data['timestep'])
        max_timesteps = 10 # can be changed 
        timesteps = timesteps[:max_timesteps]
        max_particles_per_step = 1000
        # normalise axis for plot
        axes = ThreeDAxes(
            x_range=[0, 1, 0.1],  
            y_range=[0, 1, 0.1],
            z_range=[0, 1, 0.1],
            axis_config={"include_tip": True}
        )
        self.add(axes)

        for timestep_idx, timestep in enumerate(timesteps, start=1):
            timestep_data = data[data['timestep'] == timestep]
            timestep_data = timestep_data.head(max_particles_per_step)
            points = [
                Dot3D([row['x'], row['y'], row['z']], radius=0.01) 
                for _, row in timestep_data.iterrows()
            ]
            group = VGroup(*points)
            print("Animating particles")
            self.play(FadeIn(group), run_time=0.5)
            self.wait(0.5)
            self.play(FadeOut(group), run_time=0.5)
            print("Animation complete")

        print("Rendering finished")