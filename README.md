# Molecule-Diffusion-Sim
A program that simulates how molecules diffuse over time

This fork of the project uses CuPy to make the calculations faster. This is done with the GPU. Do note, CuPy uses NVIDIA GPU's with CUDA specifically. When using this version of the program, make sure to have CuPy using the correct version of CUDA.

The code can be edited to use PyTorch instead to correctly use the GPU for computers that don't have NVIDIA GPU's
