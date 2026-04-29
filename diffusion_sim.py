import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

class diffusionSim:
    def __init__(self, num_samples=256, step_sizes=None, simple_movement=True, colors=None, molecule_ids=None):
        self.num_samples = num_samples
        self.step_sizes = cp.array(step_sizes) if step_sizes else cp.ones(num_samples)
        self.simple_movement = simple_movement
        self.colors = np.array(colors) if colors else np.array(["white"] * num_samples)
        self.molecule_ids = np.array(molecule_ids) if molecule_ids is not None else np.zeros(num_samples)
        
        # Shape: (num_samples, 2). Column 0 is X, Column 1 is Y.
        # Starts all molecules at (0, 0)
        self.positions = cp.zeros((num_samples, 2))
    
    @classmethod
    def from_moleculeTypeData(cls, molecules_type_data: list[moleculeTypeData], num_molecule_types: dict[int, int], simple_movement=True):
        """
        molecules_type_data: A list of moleculeTypeData classes
        num_molecule_types: A dictionary with the molecule_id's as the key, and the number of each molecule as the value
        """
        molecule_id_to_data: dict[int, moleculeTypeData] = {}

        for molecule_type_data in molecules_type_data:
            molecule_id_to_data[molecule_type_data.molecule_id] = molecule_type_data
        
        diffusion_sim = cls(sum(num_molecule_types.values()), 0, simple_movement)
        step_sizes = []
        colors = []
        molecule_ids = []

        for molecule_id in num_molecule_types:
            if molecule_id not in molecule_id_to_data:
                print("A molecule id is given a number of when the data for that molecule type doesn't exist")
                continue

            count = num_molecule_types[molecule_id]

            step_sizes.extend([molecule_id_to_data[molecule_id].step_size] * count)
            colors.extend([molecule_id_to_data[molecule_id].color] * count)
            molecule_ids.extend([molecule_id] * count)

        diffusion_sim.step_sizes = cp.array(step_sizes)
        diffusion_sim.colors = np.array(colors)
        diffusion_sim.molecule_ids = np.array(molecule_ids)
        
        return diffusion_sim

    def time_step(self, steps=1):
        for _ in range(steps):
            if self.simple_movement:
                # Generate random integers 0, 1, 2, or 3 for each molecule
                directions = cp.random.randint(0, 4, size=self.num_samples)
                
                # Create arrays for changes in X (dx) and Y (dy)
                dx = cp.zeros(self.num_samples)
                dy = cp.zeros(self.num_samples)
                
                # Map directions to steps
                dx[directions == 0] -= self.step_sizes[directions == 0] # Left
                dy[directions == 1] -= self.step_sizes[directions == 1] # Up
                dx[directions == 2] += self.step_sizes[directions == 2] # Right
                dy[directions == 3] += self.step_sizes[directions == 3] # Down
                
                # Update all positions at once
                self.positions[:, 0] += dx
                self.positions[:, 1] += dy
                
            else:
                # Complex movement (random angle)
                # Generate a random angle for each molecule simultaneously
                angles = cp.random.uniform(0, 2 * np.pi, size=self.num_samples)
                
                # Calculate X and Y steps using trigonometry
                self.positions[:, 0] += cp.cos(angles) * self.step_sizes
                self.positions[:, 1] += cp.sin(angles) * self.step_sizes
            
            # Round all coordinates at once to 2 decimal places
            self.positions = cp.round(self.positions, 2)

    def get_displacements(self):
        # Calculate x^2 + y^2 for all molecules
        squared_positions = self.positions ** 2
        
        # Sum along the rows (axis=1) to get (x^2 + y^2), then take the square root
        displacements = cp.sqrt(cp.sum(squared_positions, axis=1))
        
        # Round the final displacements
        return cp.round(displacements, 2)

    def get_displacement_count(self):
        displacements = self.get_displacements()
        
        # np.unique counts occurrences of each unique value in the array automatically
        unique_vals, counts = cp.unique(displacements, return_counts=True)
        
        # Convert it back to a dictionary for your Tkinter/Matplotlib code
        return dict(zip(unique_vals.get(), counts.get()))

    def plot_displacement_distribution(self):
        displacements = self.get_displacements().get()
        positions_cpu = self.positions.get()

        plt.figure(figsize=(15, 5))

        # Plot 1: Displacement (Use fewer bins to smooth out irregular square root values)
        plt.subplot(1, 3, 1)
        # Dropped bins to 30 so the irregularly spaced numbers group together nicely
        plt.hist(displacements, bins=30, color='orange', rwidth=1.0)
        plt.xlabel("Displacement Distance")
        plt.ylabel("Count")
        plt.title("Distance from Origin (Rayleigh)")

        # Plot 2: X Position (Calculate exact bins for perfect integers)
        plt.subplot(1, 3, 2)
        # Create bins that span exactly from min-0.5 to max+0.5, stepping by 1
        x_min, x_max = np.min(positions_cpu[:, 0]), np.max(positions_cpu[:, 0])
        x_bins = np.arange(x_min - 0.5, x_max + 1.5, 1)
        
        plt.hist(self.positions[:, 0], bins=x_bins, color='blue', rwidth=1.0)
        plt.xlabel("X Position")
        plt.ylabel("Count")
        plt.title("X-Axis Position (Normal)")

        # Plot 3: Y Position (Calculate exact bins for perfect integers)
        plt.subplot(1, 3, 3)
        y_min, y_max = np.min(positions_cpu[:, 1]), np.max(positions_cpu[:, 1])
        y_bins = np.arange(y_min - 0.5, y_max + 1.5, 1)
        
        plt.hist(positions_cpu[:, 1], bins=y_bins, color='green', rwidth=1.0) 
        plt.xlabel("Y Position")
        plt.ylabel("Count")
        plt.title("Y-Axis Position (Normal)")

        plt.tight_layout()
        plt.show()

class moleculeTypeData:
    def __init__(self, molecule_id, step_size=1, color="white"):
        self.color = color
        self.step_size = step_size
        self.molecule_id = molecule_id

if __name__ == "__main__":
    diffusion_sim = diffusionSim(100000, simple_movement=True)

    diffusion_sim.time_step(1000)

    diffusion_sim.plot_displacement_distribution()