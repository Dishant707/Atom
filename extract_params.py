
import torch
import numpy as np
import sys

model_path = "/Users/dishant/Desktop/Atom/ani2xr.pt"
print(f"Loading {model_path}...")
state = torch.load(model_path, map_location='cpu')

def print_tensor(name, key):
    if key in state:
        t = state[key]
        print(f"\n--- {name} ---")
        print(f"Shape: {t.shape}")
        # Print valid Rust-friendly array
        data = t.numpy().flatten()
        print(f"Data: {list(data)}")
    else:
        print(f"\nMISSING: {key}")

# Repulsion XTB
print_tensor("Repulsion Y_ab", "potentials.repulsion_xtb.y_ab")
print_tensor("Repulsion SqrtAlpha_ab", "potentials.repulsion_xtb.sqrt_alpha_ab")
print_tensor("Repulsion K_rep_ab", "potentials.repulsion_xtb.k_rep_ab")

# Check if atomic numbers are stored to map the indices
print_tensor("Atomic Numbers", "atomic_numbers")
