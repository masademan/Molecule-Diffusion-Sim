import tkinter as tk
from diffusion_graphics import DiffusionGUI

if __name__ == "__main__":
    root = tk.Tk()
    app = DiffusionGUI(root)
    root.mainloop()