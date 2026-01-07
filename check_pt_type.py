
import torch
import sys

model_path = "/Users/dishant/Desktop/Atom/ani2xr.pt"

print(f"Checking {model_path}...")

try:
    # Try loading as TorchScript
    model = torch.jit.load(model_path)
    print("SUCCESS: Model is a TorchScript file.")
    # Check graph for outputs
    print(model.graph)
    sys.exit(0)
except Exception as e:
    print(f"Not TorchScript: {e}")

try:
    # Try loading as state_dict
    state = torch.load(model_path, map_location='cpu')
    if isinstance(state, dict):
        print("SUCCESS: Model is a state_dict (dictionary of weights).")
        print("Keys:", state.keys())
    else:
        print(f"Loaded object is {type(state)}")
except Exception as e:
    print(f"Failed to load as pickle: {e}")
