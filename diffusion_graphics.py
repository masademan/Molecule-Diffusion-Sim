import time
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import tkinter.messagebox as mb
import matplotlib.colors as mcolors
from diffusion_sim import diffusionSim, moleculeTypeData
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.id = None
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hide()

    def schedule(self):
        self.unschedule()
        # Wait 500 milliseconds before showing the popup
        self.id = self.widget.after(500, self.show) 

    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def show(self, event=None):
        x = y = 0
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        
        # Create a top-level window without borders
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")
        
        label = tk.Label(self.tooltip_window, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                         font=("Arial", 9, "normal"))
        label.pack(ipadx=2, ipady=2)

    def hide(self):
        tw = self.tooltip_window
        self.tooltip_window = None
        if tw:
            tw.destroy()

class DiffusionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive Diffusion Simulation")
        self.root.geometry("1400x900")

        self.history = None 
        self.current_time_step = 0
        self.global_max = 10
        self.molecule_rows = []
        self.row_counter = 1
        self.first_sim = True

        self.setup_ui()
        self.run_simulation()

    def setup_ui(self):
        control_frame = tk.Frame(self.root, width=450, bg="#f0f0f0", padx=10, pady=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(control_frame, text="Simulation Controls", font=("Arial", 14, "bold"), bg="#f0f0f0").pack(pady=10)

        # --- TIME BOUNDS (Start & End Points) ---
        time_frame = tk.Frame(control_frame, bg="#f0f0f0")
        time_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(time_frame, text="Start Time:", bg="#f0f0f0").grid(row=0, column=0, sticky="w")
        self.start_time_var = tk.IntVar(value=0)
        tk.Entry(time_frame, textvariable=self.start_time_var, width=8).grid(row=0, column=1, padx=5, pady=2)

        tk.Label(time_frame, text="End Time:", bg="#f0f0f0").grid(row=0, column=2, sticky="w")
        self.end_time_var = tk.IntVar(value=100)
        tk.Entry(time_frame, textvariable=self.end_time_var, width=8).grid(row=0, column=3, padx=5, pady=2)
        
        tk.Label(time_frame, text="Compute Timeout (s):", bg="#f0f0f0").grid(row=1, column=0, columnspan=2, sticky="w", pady=(5,0))
        self.timeout_var = tk.DoubleVar(value=5.0) # Defaults to stopping if it takes over 5 seconds
        tk.Entry(time_frame, textvariable=self.timeout_var, width=8).grid(row=1, column=2, sticky="w", padx=5, pady=(5,0))

        # --- MOVEMENT SETTINGS ---
        movement_frame = tk.Frame(control_frame, bg="#f0f0f0")
        movement_frame.pack(fill=tk.X, pady=5)
        
        self.simple_movement_var = tk.BooleanVar(value=True)
        movement_checkbox = tk.Checkbutton(movement_frame, text="Simple Movement (Grid)", variable=self.simple_movement_var, bg="#f0f0f0")
        movement_checkbox.pack(side=tk.LEFT)
        
        # Attach the Tooltip to the checkbox!
        ToolTip(movement_checkbox, "Checked: Molecules only step Up, Down, Left, or Right on a grid.\nUnchecked: Molecules can step in any continuous random angle.")

        # --- DYNAMIC MOLECULE MENU (GRID LAYOUT) ---
        menu_label_frame = tk.Frame(control_frame, bg="#f0f0f0")
        menu_label_frame.pack(fill=tk.X, pady=(15, 5))
        
        tk.Label(menu_label_frame, text="Molecule Types", font=("Arial", 12, "bold"), bg="#f0f0f0").pack(side=tk.LEFT)
        tk.Button(menu_label_frame, text="+ Add Type", command=self.add_molecule_row).pack(side=tk.RIGHT)

        # The container that will hold both headers AND inputs using the grid system
        self.molecule_container = tk.Frame(control_frame, bg="#f0f0f0")
        self.molecule_container.pack(fill=tk.X, pady=5)
        
        # Create headers at row 0 of the grid
        headers = ["Show", "ID", "Count", "Step", "Color", "Intensity", ""]
        for col, text in enumerate(headers):
            tk.Label(self.molecule_container, text=text, bg="#f0f0f0", font=("Arial", 9, "bold")).grid(row=0, column=col, padx=2, pady=2)

        # Populate defaults
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
        # We assign a row index for the grid layout
        current_row = self.row_counter
        self.row_counter += 1

        if default_id is None:
            default_id = len(self.molecule_rows) + 1

        show_var = tk.BooleanVar(value=True)
        id_var = tk.IntVar(value=default_id)
        count_var = tk.IntVar(value=default_count)
        step_var = tk.DoubleVar(value=default_step)
        color_var = tk.StringVar(value=default_color)
        intensity_var = tk.DoubleVar(value=default_intensity)

        trigger_update = lambda *args: self.update_visuals(self.time_slider.get())
        show_var.trace_add('write', trigger_update)
        intensity_var.trace_add('write', trigger_update)

        # Place widgets directly into the grid container
        chk = tk.Checkbutton(self.molecule_container, variable=show_var, bg="#f0f0f0")
        chk.grid(row=current_row, column=0, pady=2)
        
        ent_id = tk.Entry(self.molecule_container, textvariable=id_var, width=4)
        ent_id.grid(row=current_row, column=1, pady=2)
        
        ent_count = tk.Entry(self.molecule_container, textvariable=count_var, width=6)
        ent_count.grid(row=current_row, column=2, pady=2)
        
        ent_step = tk.Entry(self.molecule_container, textvariable=step_var, width=6)
        ent_step.grid(row=current_row, column=3, pady=2)
        
        cb_color = ttk.Combobox(self.molecule_container, textvariable=color_var, width=8, 
                                values=["blue", "red", "green", "orange", "purple", "cyan", "black", "magenta"])
        cb_color.grid(row=current_row, column=4, pady=2)
        
        ent_int = tk.Entry(self.molecule_container, textvariable=intensity_var, width=6)
        ent_int.grid(row=current_row, column=5, pady=2)

        def remove_row():
            # Destroying widgets in a grid collapses that row automatically!
            chk.destroy()
            ent_id.destroy()
            ent_count.destroy()
            ent_step.destroy()
            cb_color.destroy()
            ent_int.destroy()
            btn_del.destroy()
            
            # Remove from our data list
            self.molecule_rows = [row for row in self.molecule_rows if row["id_widget"] != ent_id]
            self.update_visuals(self.time_slider.get())
            
        btn_del = tk.Button(self.molecule_container, text="X", fg="red", command=remove_row, padx=2, pady=0)
        btn_del.grid(row=current_row, column=6, padx=2, pady=2)

        # Store references
        self.molecule_rows.append({
            "id_widget": ent_id, # Used to safely identify this row during deletion
            "show": show_var, 
            "id": id_var, 
            "count": count_var, 
            "step": step_var, 
            "color": color_var, 
            "intensity": intensity_var
        })

    def run_simulation(self):
        print("Reading menu and compiling simulation...")
        try:
            start_time = self.start_time_var.get()
            end_time = self.end_time_var.get()
            max_timeout = float(self.timeout_var.get())
            time_steps = end_time - start_time + 1
            if time_steps <= 0: time_steps = 1
        except ValueError:
            mb.showerror("Simulation Error", "Error: Start and End times must be numbers!")
            # print("Error: Start and End times must be numbers!")
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
                mb.showerror("Simulation Error", "Error reading a row. Make sure inputs are numbers!")
                # print("Error reading a row. Make sure inputs are numbers!")
                return
            
        is_simple = self.simple_movement_var.get()
        # self.sim = diffusionSim.from_moleculeTypeData(molecules_type_data, num_molecule_types, simple_movement=is_simple)
        # self.history = np.zeros((time_steps, total_samples, 2))

        try:
            self.sim = diffusionSim.from_moleculeTypeData(molecules_type_data, num_molecule_types, simple_movement=is_simple)
        except:
            mb.showerror("Memory Error", f"Too many molecules are being simulated, try decreasing the number of molecules")
            return

        try:
            self.history = np.zeros((time_steps, total_samples, 2))
        except:
            mb.showerror("Memory Error", f"Could not allocate memory for matrix of shape ({time_steps}, {total_samples}, 2).\nTry lowering the number of molecules or time frame size")
            return

        self.sim.time_step(start_time)
        self.history[0] = self.sim.positions.copy()

        # --- TIMEOUT SAFETY LOOP ---
        start_compute_time = time.time()
        actual_completed_steps = 1
        
        for t in range(time_steps - 1):
            # Check the clock every single frame to see if we've exceeded the limit
            if time.time() - start_compute_time > max_timeout:
                mb.showerror("Simulation Warning", f"Warning: Simulation cut short! Reached compute timeout of {max_timeout}s.")
                # print(f"Warning: Simulation cut short! Reached compute timeout of {max_timeout}s.")
                # Chop off the empty zeros at the end of the history array
                self.history = self.history[:actual_completed_steps]
                break

            self.sim.time_step(1)
            self.history[t + 1] = self.sim.positions.copy()
            actual_completed_steps += 1
        
        if actual_completed_steps == 0:
            mb.showerror("Simulation Error", "Simulation failed or timed out immediately.")
            # print("Simulation failed or timed out immediately.")
            return
        
        if self.history.shape == (time_steps, total_samples, 2) and not self.first_sim:
            mb.showinfo("Simulation finished", "The simulation has finished succesfully")
        self.first_sim = False

        all_x = self.history[:, :, 0]
        all_y = self.history[:, :, 1]
        all_distances = np.sqrt(all_x**2 + all_y**2)
        self.max_simulation_dist = np.max(all_distances) if len(all_distances) > 0 else 10

        unique_ids = np.unique(self.sim.molecule_ids)
        self.hist_filter_dropdown["values"] = ["All"] + [f"Type {m_id}" for m_id in unique_ids]
        self.hist_filter_var.set("All")

        self.time_slider.config(to=max(0, actual_completed_steps - 1))
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

        map_bound = self.max_simulation_dist + self.radius_slider.get() * 1.5
        
        self.ax_map.set_xlim(-map_bound, map_bound)
        self.ax_map.set_ylim(-map_bound, map_bound)

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

        disp_bins = np.linspace(0, map_bound, 31)
        xy_bins = np.linspace(-map_bound, map_bound, 31)

        if len(hist_distances) > 0:
            self.ax_hist_disp.hist(hist_distances, bins=disp_bins, color=c_disp, rwidth=1.0)
        self.ax_hist_disp.set_title(f"Displacement ({filter_val})")
        self.ax_hist_disp.set_xlim(0, map_bound)

        if len(hist_x_coords) > 0:
            self.ax_hist_x.hist(hist_x_coords, bins=xy_bins, color=c_x, rwidth=1.0)
        self.ax_hist_x.set_title(f"X Position ({filter_val})")
        self.ax_hist_x.set_xlim(-map_bound, map_bound)

        if len(hist_y_coords) > 0:
            self.ax_hist_y.hist(hist_y_coords, bins=xy_bins, color=c_y, rwidth=1.0)
        self.ax_hist_y.set_title(f"Y Position ({filter_val})")
        self.ax_hist_y.set_xlim(-map_bound, map_bound)

        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = DiffusionGUI(root)
    root.mainloop()