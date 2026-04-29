import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from diffusion_sim import diffusionSim, moleculeTypeData
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from diffusion_sim import diffusionSim, moleculeTypeData
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class DiffusionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive Diffusion Simulation")
        self.root.geometry("1400x900")

        self.history = None 
        self.current_time_step = 0
        self.global_max = 10
        self.molecule_rows = []

        self.setup_ui()
        self.run_simulation()

    def setup_ui(self):
        control_frame = tk.Frame(self.root, width=450, bg="#f0f0f0", padx=10, pady=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(control_frame, text="Simulation Controls", font=("Arial", 14, "bold"), bg="#f0f0f0").pack(pady=10)

        # --- TIME BOUNDS (Start & End Points) ---
        time_frame = tk.Frame(control_frame, bg="#f0f0f0")
        time_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(time_frame, text="Start Time:", bg="#f0f0f0").pack(side=tk.LEFT)
        self.start_time_var = tk.DoubleVar(value=0.0)
        tk.Entry(time_frame, textvariable=self.start_time_var, width=8).pack(side=tk.LEFT, padx=5)

        tk.Label(time_frame, text="End Time:", bg="#f0f0f0").pack(side=tk.LEFT)
        self.end_time_var = tk.DoubleVar(value=100.0)
        tk.Entry(time_frame, textvariable=self.end_time_var, width=8).pack(side=tk.LEFT, padx=5)

        # --- DYNAMIC MOLECULE MENU ---
        menu_label_frame = tk.Frame(control_frame, bg="#f0f0f0")
        menu_label_frame.pack(fill=tk.X, pady=(15, 5))
        
        tk.Label(menu_label_frame, text="Molecule Types", font=("Arial", 12, "bold"), bg="#f0f0f0").pack(side=tk.LEFT)
        tk.Button(menu_label_frame, text="+ Add Type", command=self.add_molecule_row).pack(side=tk.RIGHT)

        header_frame = tk.Frame(control_frame, bg="#f0f0f0")
        header_frame.pack(fill=tk.X)
        
        # Updated headers
        tk.Label(header_frame, text="Show", width=4, bg="#f0f0f0").pack(side=tk.LEFT, padx=1)
        tk.Label(header_frame, text="ID", width=3, bg="#f0f0f0").pack(side=tk.LEFT, padx=1)
        tk.Label(header_frame, text="Count", width=6, bg="#f0f0f0").pack(side=tk.LEFT, padx=1)
        tk.Label(header_frame, text="Step", width=5, bg="#f0f0f0").pack(side=tk.LEFT, padx=1)
        tk.Label(header_frame, text="Color", width=8, bg="#f0f0f0").pack(side=tk.LEFT, padx=1)
        tk.Label(header_frame, text="Intensity", width=6, bg="#f0f0f0").pack(side=tk.LEFT, padx=1)

        self.molecule_container = tk.Frame(control_frame, bg="#f0f0f0")
        self.molecule_container.pack(fill=tk.X, pady=5)
        
        self.add_molecule_row(default_id=1, default_count=200, default_step=1.0, default_color="blue", default_intensity=1.0)
        self.add_molecule_row(default_id=2, default_count=200, default_step=2.0, default_color="red", default_intensity=1.5)

        # --- SLIDERS & BUTTONS ---
        tk.Label(control_frame, text="Molecule Render Radius", bg="#f0f0f0").pack(pady=(15, 0))
        self.radius_slider = tk.Scale(control_frame, from_=1, to=20, orient=tk.HORIZONTAL, bg="#f0f0f0", command=lambda e: self.update_visuals(self.time_slider.get()))
        self.radius_slider.set(5)
        self.radius_slider.pack(fill=tk.X)

        tk.Button(control_frame, text="Run & Pre-compute Simulation", command=self.run_simulation, bg="#d0ffd0", font=("Arial", 10, "bold")).pack(pady=15, fill=tk.X)

        tk.Label(control_frame, text="Time Scrubber", bg="#f0f0f0").pack(pady=(5, 0))
        self.time_slider = tk.Scale(control_frame, from_=0, to=100, orient=tk.HORIZONTAL, bg="#f0f0f0", command=self.update_visuals)
        self.time_slider.pack(fill=tk.X)

        # --- HISTOGRAM FILTER MENU ---
        tk.Label(control_frame, text="Histogram Filter", bg="#f0f0f0").pack(pady=(15, 0))
        self.hist_filter_var = tk.StringVar(value="All")
        self.hist_filter_dropdown = ttk.Combobox(control_frame, textvariable=self.hist_filter_var, state="readonly")
        self.hist_filter_dropdown.pack(fill=tk.X)
        self.hist_filter_dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_visuals(self.time_slider.get()))

        # --- Right Panel: Visuals ---
        visual_frame = tk.Frame(self.root, bg="white")
        visual_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = plt.Figure(figsize=(10, 8))
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1, hspace=0.3, wspace=0.25)
        
        self.ax_map = self.fig.add_subplot(2, 3, (1, 3)) 
        self.ax_hist_disp = self.fig.add_subplot(2, 3, 4)
        self.ax_hist_x = self.fig.add_subplot(2, 3, 5)
        self.ax_hist_y = self.fig.add_subplot(2, 3, 6)

        self.canvas = FigureCanvasTkAgg(self.fig, master=visual_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def add_molecule_row(self, default_id=None, default_count=100, default_step=1.0, default_color="blue", default_intensity=1.0):
        row_frame = tk.Frame(self.molecule_container, bg="#f0f0f0")
        row_frame.pack(fill=tk.X, pady=2)

        if default_id is None:
            default_id = len(self.molecule_rows) + 1

        show_var = tk.BooleanVar(value=True)
        id_var = tk.IntVar(value=default_id)
        count_var = tk.IntVar(value=default_count)
        step_var = tk.DoubleVar(value=default_step)
        color_var = tk.StringVar(value=default_color)
        intensity_var = tk.DoubleVar(value=default_intensity)

        # Auto-update graph instantly when checking boxes or changing intensity
        trigger_update = lambda *args: self.update_visuals(self.time_slider.get())
        show_var.trace_add('write', trigger_update)
        intensity_var.trace_add('write', trigger_update)

        tk.Checkbutton(row_frame, variable=show_var, bg="#f0f0f0").pack(side=tk.LEFT, padx=1)
        tk.Entry(row_frame, textvariable=id_var, width=3).pack(side=tk.LEFT, padx=1)
        tk.Entry(row_frame, textvariable=count_var, width=6).pack(side=tk.LEFT, padx=1)
        tk.Entry(row_frame, textvariable=step_var, width=5).pack(side=tk.LEFT, padx=1)
        
        color_dropdown = ttk.Combobox(row_frame, textvariable=color_var, width=8, 
                                      values=["blue", "red", "green", "orange", "purple", "cyan", "black", "magenta"])
        color_dropdown.pack(side=tk.LEFT, padx=1)
        
        tk.Entry(row_frame, textvariable=intensity_var, width=5).pack(side=tk.LEFT, padx=1)

        def remove_row():
            row_frame.destroy()
            self.molecule_rows = [row for row in self.molecule_rows if row["frame"] != row_frame]
            self.update_visuals(self.time_slider.get())
            
        tk.Button(row_frame, text="X", fg="red", command=remove_row, padx=2, pady=0).pack(side=tk.LEFT, padx=2)

        self.molecule_rows.append({
            "frame": row_frame, "show": show_var, "id": id_var, "count": count_var, 
            "step": step_var, "color": color_var, "intensity": intensity_var
        })

    def run_simulation(self):
        print("Reading menu and compiling simulation...")
        try:
            start_time = float(self.start_time_var.get())
            end_time = float(self.end_time_var.get())
            time_steps = int(end_time - start_time)
            if time_steps <= 0: time_steps = 1
        except ValueError:
            print("Error: Start and End times must be numbers!")
            return

        molecules_type_data = []
        num_molecule_types = {}
        total_samples = 0

        for row in self.molecule_rows:
            try:
                m_id = row["id"].get()
                count = row["count"].get()
                step = row["step"].get()
                color = row["color"].get()

                m_data = moleculeTypeData(m_id, step, color)
                molecules_type_data.append(m_data)
                num_molecule_types[m_id] = count
                total_samples += count
            except tk.TclError:
                print("Error reading a row. Make sure inputs are numbers!")
                return

        self.sim = diffusionSim.from_moleculeTypeData(molecules_type_data, num_molecule_types, simple_movement=True)
        self.history = np.zeros((time_steps, total_samples, 2))
        
        for t in range(time_steps):
            self.history[t] = self.sim.positions.copy()
            self.sim.time_step(1)

        self.global_max = np.max(np.abs(self.history)) + 2

        # Update filter dropdown
        unique_ids = np.unique(self.sim.molecule_ids)
        self.hist_filter_dropdown["values"] = ["All"] + [f"Type {m_id}" for m_id in unique_ids]
        self.hist_filter_var.set("All")

        self.time_slider.config(to=time_steps - 1)
        self.time_slider.set(0)
        self.update_visuals(0)

    def update_visuals(self, val):
        if self.history is None or len(self.history) == 0: return
        step = int(val)
        
        current_positions = self.history[step]
        
        # Gather live UI data for Show/Hide and Intensity
        type_visibility = {}
        type_intensities = {}
        for row in self.molecule_rows:
            try:
                m_id = row["id"].get()
                type_visibility[m_id] = row["show"].get()
                type_intensities[m_id] = row["intensity"].get()
            except:
                pass

        # Build a mask of ONLY the molecules checked as "Show"
        visible_mask = np.array([type_visibility.get(m_id, True) for m_id in self.sim.molecule_ids])
        
        # Apply the mask
        vis_x = current_positions[:, 0][visible_mask]
        vis_y = current_positions[:, 1][visible_mask]
        vis_ids = self.sim.molecule_ids[visible_mask]
        vis_colors = self.sim.colors[visible_mask]
        
        distances = np.sqrt(vis_x**2 + vis_y**2)
        max_dist = np.max(distances) if len(distances) > 0 else 0.1 

        self.ax_map.clear()
        self.ax_hist_disp.clear()
        self.ax_hist_x.clear()
        self.ax_hist_y.clear()

        # --- 1. DENSITY & INTENSITY CALCULATION ---
        if len(vis_x) > 0:
            bins = 30
            H, xedges, yedges = np.histogram2d(vis_x, vis_y, bins=bins)
            x_idx = np.clip(np.digitize(vis_x, xedges) - 1, 0, bins - 1)
            y_idx = np.clip(np.digitize(vis_y, yedges) - 1, 0, bins - 1)
            
            densities = H[x_idx, y_idx]
            max_dens = np.max(densities) if np.max(densities) > 0 else 1
            
            # Look up the custom intensity multiplier for every individual molecule
            intensities_array = np.array([type_intensities.get(m_id, 1.0) for m_id in vis_ids])
            
            # Apply intensity to the density! 
            adjusted_densities = (densities / max_dens) * intensities_array
            
            # Clip between 0.15 (so it doesn't totally vanish) and 1.0 (max color)
            norm_densities = np.clip(0.15 + 0.85 * adjusted_densities, 0.15, 1.0)
            
            base_rgbs = np.array([mcolors.to_rgb(c) for c in vis_colors])
            d = norm_densities[:, np.newaxis] 
            mixed_rgbs = base_rgbs * d + np.array([1.0, 1.0, 1.0]) * (1 - d)

            # --- 2. DRAWING THE MAP ---
            marker_area = self.radius_slider.get() ** 2 
            self.ax_map.scatter(vis_x, vis_y, c=mixed_rgbs, s=marker_area, alpha=1.0, edgecolors='none')
                                
        self.ax_map.plot(0, 0, marker='+', color='red', markersize=15, markeredgewidth=2)
        circle = plt.Circle((0, 0), max_dist, color='gray', fill=False, linestyle='--')
        self.ax_map.add_patch(circle)

        # Factor in the custom start time for the label
        start_t = float(self.start_time_var.get())
        self.ax_map.set_title(f"Molecule Map (Time: {start_t + step:.1f})")
        self.ax_map.set_aspect('equal')
        self.ax_map.grid(True, linestyle=':', alpha=0.6)
        self.ax_map.set_xlim(-self.global_max, self.global_max)
        self.ax_map.set_ylim(-self.global_max, self.global_max)

        # --- 3. FILTERING & DRAWING HISTOGRAMS ---
        filter_val = self.hist_filter_var.get()
        
        if filter_val == "All":
            hist_mask = np.ones(len(vis_x), dtype=bool) 
            c_disp, c_x, c_y = 'orange', 'blue', 'green'
        else:
            type_id = int(filter_val.split(" ")[1])
            hist_mask = vis_ids == type_id
            type_color = vis_colors[hist_mask][0] if len(vis_colors[hist_mask]) > 0 else 'gray'
            c_disp = c_x = c_y = type_color

        hist_distances = distances[hist_mask]
        hist_x_coords = vis_x[hist_mask]
        hist_y_coords = vis_y[hist_mask]

        if len(hist_distances) > 0:
            self.ax_hist_disp.hist(hist_distances, bins=30, color=c_disp, rwidth=1.0)
        self.ax_hist_disp.set_title(f"Displacement ({filter_val})")
        self.ax_hist_disp.set_xlim(0, self.global_max)

        # FIX: Switched np.arange() bins out for bins=30 to eliminate gaps/aliasing!
        if len(hist_x_coords) > 0:
            self.ax_hist_x.hist(hist_x_coords, bins=30, color=c_x, rwidth=1.0)
        self.ax_hist_x.set_title(f"X Position ({filter_val})")
        self.ax_hist_x.set_xlim(-self.global_max, self.global_max)

        if len(hist_y_coords) > 0:
            self.ax_hist_y.hist(hist_y_coords, bins=30, color=c_y, rwidth=1.0)
        self.ax_hist_y.set_title(f"Y Position ({filter_val})")
        self.ax_hist_y.set_xlim(-self.global_max, self.global_max)

        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = DiffusionGUI(root)
    root.mainloop()