import time
import random
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

        self.prev_start_time = 0
        self.prev_end_time = 100
        
        self.last_physics_state = None

        self.is_playing = False
        self.play_job = None

        self.setup_ui()
        self.run_simulation()

    def setup_ui(self):
        control_frame = tk.Frame(self.root, width=450, padx=10, pady=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(control_frame, text="Simulation Controls", font=("Arial", 14, "bold")).pack(pady=10)

        # --- TIME BOUNDS (Start & End Points) ---
        time_frame = tk.Frame(control_frame)
        time_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(time_frame, text="Start Time:").grid(row=0, column=0, sticky="w")
        self.start_time_var = tk.IntVar(value=0)
        tk.Entry(time_frame, textvariable=self.start_time_var, width=8).grid(row=0, column=1, padx=5, pady=2)

        tk.Label(time_frame, text="End Time:").grid(row=0, column=2, sticky="w")
        self.end_time_var = tk.IntVar(value=100)
        tk.Entry(time_frame, textvariable=self.end_time_var, width=8).grid(row=0, column=3, padx=5, pady=2)
        
        tk.Label(time_frame, text="Timeout (s):").grid(row=1, column=0, sticky="w", pady=(5,0))
        self.timeout_var = tk.DoubleVar(value=5.0) # Defaults to stopping if it takes over 5 seconds
        tk.Entry(time_frame, textvariable=self.timeout_var, width=8).grid(row=1, column=1, sticky="w", padx=5, pady=(5,0))

        # SEED UI COMPONENT
        tk.Label(time_frame, text="Seed (-1=Rand):").grid(row=1, column=2, sticky="w", pady=(5,0))
        self.seed_var = tk.IntVar(value=-1)
        tk.Entry(time_frame, textvariable=self.seed_var, width=8).grid(row=1, column=3, sticky="w", padx=5, pady=(5,0))

        # --- MOVEMENT SETTINGS ---
        movement_frame = tk.Frame(control_frame)
        movement_frame.pack(fill=tk.X, pady=5)
        
        self.simple_movement_var = tk.BooleanVar(value=True)
        movement_checkbox = tk.Checkbutton(movement_frame, text="Simple Movement (Grid)", variable=self.simple_movement_var)
        movement_checkbox.pack(side=tk.LEFT)
        
        # Attach the Tooltip to the checkbox!
        ToolTip(movement_checkbox, "Checked: Molecules only step Up, Down, Left, or Right on a grid.\nUnchecked: Molecules can step in any continuous random angle.")

        # --- DYNAMIC MOLECULE MENU (GRID LAYOUT) ---
        menu_label_frame = tk.Frame(control_frame)
        menu_label_frame.pack(fill=tk.X, pady=(15, 5))
        
        tk.Label(menu_label_frame, text="Molecule Types", font=("Arial", 12, "bold")).pack(side=tk.LEFT)
        tk.Button(menu_label_frame, text="+ Add Type", command=self.add_molecule_row).pack(side=tk.RIGHT)

        # The container that will hold both headers AND inputs using the grid system
        self.molecule_container = tk.Frame(control_frame)
        self.molecule_container.pack(fill=tk.X, pady=5)
        
        # Create headers at row 0 of the grid
        headers = ["Show", "ID", "Count", "Step", "Color", "Intensity", ""]
        for col, text in enumerate(headers):
            tk.Label(self.molecule_container, text=text, font=("Arial", 9, "bold")).grid(row=0, column=col, padx=2, pady=2)

        # Populate defaults
        self.add_molecule_row(default_id=1, default_count=200, default_step=1.0, default_color="blue", default_intensity=1.0)
        self.add_molecule_row(default_id=2, default_count=200, default_step=2.0, default_color="red", default_intensity=1.5)
        
        # --- SLIDERS & BUTTONS ---
        tk.Label(control_frame, text="Molecule Render Radius").pack(pady=(15, 0))
        self.radius_slider = tk.Scale(control_frame, from_=1, to=20, orient=tk.HORIZONTAL, command=lambda e: self.update_visuals(self.time_slider.get()))
        self.radius_slider.set(5)
        self.radius_slider.pack(fill=tk.X)

        tk.Button(control_frame, text="Run & Pre-compute Simulation", command=self.run_simulation, bg="#d0ffd0", font=("Arial", 10, "bold")).pack(pady=15, fill=tk.X)

        tk.Label(control_frame, text="Time Scrubber").pack(pady=(5, 0))
        self.time_slider = tk.Scale(control_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.update_visuals)
        self.time_slider.pack(fill=tk.X)
        self.time_slider.bind("<ButtonPress-1>", self.pause_playback)

        # --- PLAYBACK CONTROLS ---
        playback_frame = tk.Frame(control_frame)
        playback_frame.pack(fill=tk.X, pady=(5, 10))
        
        self.play_button = tk.Button(playback_frame, text="▶ Play", width=8, command=self.toggle_play, bg="#d0d0ff", font=("Arial", 9, "bold"))
        self.play_button.pack(side=tk.LEFT, padx=2)
        
        tk.Label(playback_frame, text="Sec/Frame:").pack(side=tk.LEFT, padx=(10, 2))
        self.playback_speed_var = tk.DoubleVar(value=0.1) 
        tk.Entry(playback_frame, textvariable=self.playback_speed_var, width=5).pack(side=tk.LEFT, padx=2)
        
        self.loop_var = tk.BooleanVar(value=True)
        tk.Checkbutton(playback_frame, text="Loop", variable=self.loop_var).pack(side=tk.LEFT, padx=(10, 2))

        # --- VISUAL TOGGLES ---
        visuals_toggle_frame = tk.Frame(control_frame)
        visuals_toggle_frame.pack(fill=tk.X, pady=(15, 0))
        tk.Label(visuals_toggle_frame, text="Visual Toggles", font=("Arial", 10, "bold")).pack(anchor="w")

        self.show_grid_var = tk.BooleanVar(value=True)
        self.show_cross_var = tk.BooleanVar(value=True)
        self.show_circle_var = tk.BooleanVar(value=True)

        trigger_visuals = lambda *args: self.update_visuals(self.time_slider.get())
        self.show_grid_var.trace_add('write', trigger_visuals)
        self.show_cross_var.trace_add('write', trigger_visuals)
        self.show_circle_var.trace_add('write', trigger_visuals)

        tk.Checkbutton(visuals_toggle_frame, text="Grid", variable=self.show_grid_var).pack(side=tk.LEFT)
        tk.Checkbutton(visuals_toggle_frame, text="Origin Cross", variable=self.show_cross_var).pack(side=tk.LEFT)
        tk.Checkbutton(visuals_toggle_frame, text="Max Radius", variable=self.show_circle_var).pack(side=tk.LEFT)

        # --- HISTOGRAM FILTER MENU ---
        tk.Label(control_frame, text="Histogram Filter").pack(pady=(15, 0))
        self.hist_filter_var = tk.StringVar(value="All")
        self.hist_filter_dropdown = ttk.Combobox(control_frame, textvariable=self.hist_filter_var, state="readonly")
        self.hist_filter_dropdown.pack(fill=tk.X)
        self.hist_filter_dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_visuals(self.time_slider.get()))

        # --- Right Panel: Visuals ---
        visual_frame = tk.Frame(self.root)
        visual_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = plt.Figure(figsize=(10, 8))
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1, hspace=0.3, wspace=0.25)
        
        self.ax_map = self.fig.add_subplot(2, 3, (1, 3)) 
        self.ax_hist_disp = self.fig.add_subplot(2, 3, 4)
        self.ax_hist_x = self.fig.add_subplot(2, 3, 5)
        self.ax_hist_y = self.fig.add_subplot(2, 3, 6)

        self.canvas = FigureCanvasTkAgg(self.fig, master=visual_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def toggle_play(self):
        if self.history is None or len(self.history) == 0:
            return

        self.is_playing = not self.is_playing

        if self.is_playing:
            # Changes button to Pause!
            self.play_button.config(text="⏸ Pause", bg="#ffd0d0")
            if self.time_slider.get() >= len(self.history) - 1:
                self.time_slider.set(0)
            self.play_step()
        else:
            # Changes button back to Play!
            self.play_button.config(text="▶ Play", bg="#d0d0ff")
            if self.play_job:
                self.root.after_cancel(self.play_job)
                self.play_job = None
    
    def pause_playback(self, event=None):
        # Triggered the exact moment the user clicks the Time Scrubber
        if self.is_playing:
            self.toggle_play()

    def play_step(self):
        if not self.is_playing or self.history is None:
            return
        
        current_step = self.time_slider.get()
        max_step = len(self.history) - 1

        if current_step < max_step:
            self.time_slider.set(current_step + 1)
        else:
            if self.loop_var.get():
                self.time_slider.set(0) 
            else:
                self.toggle_play() 
                return
        
        try:
            sec_per_frame = float(self.playback_speed_var.get())
            if sec_per_frame <= 0.001: sec_per_frame = 0.001 
        except ValueError:
            sec_per_frame = 0.1 
        
        delay_ms = int(sec_per_frame * 1000)
        self.play_job = self.root.after(delay_ms, self.play_step)

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
        color_var.trace_add('write', trigger_update)

        # Place widgets directly into the grid container
        chk = tk.Checkbutton(self.molecule_container, variable=show_var)
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

    def get_physics_state(self):
        try:
            state = {
                "simple_movement": self.simple_movement_var.get(),
                "seed": self.seed_var.get(),
                "molecules": []
            }
            # We explicitly ignore color and intensity because they don't impact the math!
            for row in self.molecule_rows:
                state["molecules"].append((row["id"].get(), row["count"].get(), row["step"].get()))
            return str(state)
        except:
            return "error"

    def run_simulation(self):
        print("Reading menu and compiling simulation...")
        try:
            start_time = self.start_time_var.get()
            end_time = self.end_time_var.get()
            max_timeout = float(self.timeout_var.get())
            if end_time < start_time:
                mb.showerror("Parameter Error", "End time must be greater than start time.")
                return
            time_steps = end_time - start_time + 1

            # --- INTERCEPT AND GENERATE SEED ---
            if self.seed_var.get() == -1:
                # Generate a random integer seed between 0 and 999,999,999
                seed_val = random.randint(0, 9999999)
                # Overwrite the -1 in the text box instantly!
                self.seed_var.set(seed_val)
        except ValueError:
            mb.showerror("Simulation Error", "Start and End times must be numbers!")
            return
        
        current_physics_state = self.get_physics_state()
        total_samples = sum(row["count"].get() for row in self.molecule_rows)
        time_steps_to_add = 0

        if self.history is not None and self.last_physics_state == current_physics_state and start_time == self.prev_start_time and end_time == self.prev_end_time:
            mb.showinfo("Simulation finished", "The simulation has finished succesfully")
            return
        
        if self.history is not None and self.last_physics_state == current_physics_state:
            num_done = 0

            if start_time >= self.prev_start_time:
                amount_to_cut_from_beginning = start_time - self.prev_start_time
                self.history = self.history[amount_to_cut_from_beginning:]
                num_done += 1

            if end_time <= self.prev_end_time:
                amount_to_cut_from_end = len(self.history) + end_time - self.prev_end_time
                self.history = self.history[:amount_to_cut_from_end]
                num_done += 1

            if num_done == 2:
                all_x = self.history[:, :, 0]
                all_y = self.history[:, :, 1]
                all_distances = np.sqrt(all_x**2 + all_y**2)
                self.max_simulation_dist = np.max(all_distances) if len(all_distances) > 0 else 10

                unique_ids = np.unique(self.sim.molecule_ids)
                self.hist_filter_dropdown["values"] = ["All"] + [f"Type {m_id}" for m_id in unique_ids]
                self.hist_filter_var.set("All")

                self.prev_start_time = start_time
                self.prev_end_time = end_time

                self.time_slider.config(to=max(0, len(self.history) - 1))
                self.time_slider.set(0)
                self.update_visuals(0)

                mb.showinfo("Simulation finished", "The simulation has finished succesfully")
                return
        
        if self.history is not None and self.last_physics_state == current_physics_state and start_time >= self.prev_start_time and start_time < self.prev_end_time and end_time > self.prev_end_time:
            time_steps_to_add = end_time - self.prev_end_time

            try:
                history = np.zeros((time_steps_to_add, total_samples, 2))
            except:
                mb.showerror("Memory Error", f"Could not allocate memory for matrix of shape ({time_steps_to_add}, {total_samples}, 2).\nTry lowering the number of molecules or time frame size")
                return
            
            starting_steps = 0
            time_steps = time_steps_to_add
            sim = self.sim

        else:
            molecules_type_data = []
            num_molecule_types = {}

            for row in self.molecule_rows:
                try:
                    m_id = row["id"].get()
                    count = row["count"].get()
                    step = row["step"].get()
                    color = row["color"].get()

                    m_data = moleculeTypeData(m_id, step, color)
                    molecules_type_data.append(m_data)
                    num_molecule_types[m_id] = count
                except tk.TclError:
                    mb.showerror("Simulation Error", "Error reading a row. Make sure inputs are numbers!")
                    return
                
            is_simple = self.simple_movement_var.get()

            try:
                sim = diffusionSim.from_moleculeTypeData(molecules_type_data, num_molecule_types, simple_movement=is_simple, seed=self.seed_var.get())
            except:
                mb.showerror("Memory Error", f"Too many molecules are being simulated, try decreasing the number of molecules")
                return

            try:
                history = np.zeros((time_steps, total_samples, 2))
            except:
                mb.showerror("Memory Error", f"Could not allocate memory for matrix of shape ({time_steps}, {total_samples}, 2).\nTry lowering the number of molecules or time frame size")
                return

            self.sim = sim

            sim.time_step(start_time)
            history[0] = sim.positions.copy()
            starting_steps = 1


        # --- TIMEOUT SAFETY LOOP ---
        start_compute_time = time.time()
        actual_completed_steps = starting_steps
        
        for t in range(time_steps - starting_steps):
            # Check the clock every single frame to see if we've exceeded the limit
            if time.time() - start_compute_time > max_timeout:
                mb.showerror("Simulation Warning", f"Simulation cut short! Reached compute timeout of {max_timeout}s.")
                # Chop off the empty zeros at the end of the history array
                history = history[:actual_completed_steps]
                break

            sim.time_step(1)
            history[t + starting_steps] = sim.positions.copy()
            actual_completed_steps += 1

        if starting_steps == 0:
            self.history = np.concatenate((self.history, history), axis=0)
        else:
            self.history = history
        
        if actual_completed_steps == 0:
            mb.showerror("Simulation Error", "Simulation failed or timed out immediately.")
            return

        if self.history.shape == (time_steps + time_steps_to_add, total_samples, 2) and not self.first_sim:
            mb.showinfo("Simulation finished", "The simulation has finished succesfully")
        self.first_sim = False

        self.last_physics_state = current_physics_state
        self.prev_start_time = start_time
        self.prev_end_time = actual_completed_steps

        all_x = self.history[:, :, 0]
        all_y = self.history[:, :, 1]
        all_distances = np.sqrt(all_x**2 + all_y**2)
        self.max_simulation_dist = np.max(all_distances) if len(all_distances) > 0 else 10

        unique_ids = np.unique(self.sim.molecule_ids)
        self.hist_filter_dropdown["values"] = ["All"] + [f"Type {m_id}" for m_id in unique_ids]
        self.hist_filter_var.set("All")

        self.time_slider.config(to=max(0, len(self.history) - 1))
        self.time_slider.set(0)
        self.update_visuals(0)

    def update_visuals(self, val):
        if self.history is None or len(self.history) == 0: return
        step = int(val)
        
        current_positions = self.history[step]
        
        # Gather live UI data for Show/Hide and Intensity
        type_visibility = {}
        type_intensities = {}
        type_colors = {}
        for row in self.molecule_rows:
            try:
                m_id = row["id"].get()
                type_visibility[m_id] = row["show"].get()
                type_intensities[m_id] = row["intensity"].get()
                type_colors[m_id] = row["color"].get()
            except:
                pass

        # Build a mask of ONLY the molecules checked as "Show"
        visible_mask = np.array([type_visibility.get(m_id, True) for m_id in self.sim.molecule_ids])
        
        # Apply the mask
        vis_x = current_positions[:, 0][visible_mask]
        vis_y = current_positions[:, 1][visible_mask]
        vis_ids = self.sim.molecule_ids[visible_mask]

        vis_colors = vis_colors = np.array([type_colors.get(m_id, "gray") for m_id in vis_ids])
        
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
                                
        if self.show_cross_var.get():
            self.ax_map.plot(0, 0, marker='+', color='red', markersize=15, markeredgewidth=2)

        if self.show_circle_var.get():
            circle = plt.Circle((0, 0), max_dist, color='gray', fill=False, linestyle='--')
            self.ax_map.add_patch(circle)

        # Factor in the custom start time for the label
        start_t = float(self.start_time_var.get())
        self.ax_map.set_title(f"Molecule Map (Time: {start_t + step:.1f})")
        self.ax_map.set_aspect('equal')

        if self.show_grid_var.get():
            self.ax_map.grid(True, linestyle=':', alpha=0.6)
        else:
            self.ax_map.grid(False)

        map_bound = self.max_simulation_dist + 10
        
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