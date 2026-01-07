import torch
import sys

model_path = "/Users/dishant/Desktop/Atom/2023-12-03-mace-128-L1_epoch-199.model-lammps.pt"

try:
    model = torch.jit.load(model_path)
    print("Model Loaded Successfully!")
    
    # Print the code (Python source of the script)
    print("\n--- Model Code (forward) ---")
    try:
        print(model.code)
    except Exception as e:
        print(f"Failed to print code: {e}")

    # Inspect the forward method inputs
    print("\n--- Forward Inputs ---")
    for arg in model.forward.schema.arguments:
        print(f"Name: {arg.name}, Type: {arg.type}")

except Exception as e:
    print(f"Error loading model: {e}")
