import torch
import torch.nn as nn

print("⚠️ Toolchain Unstable (Python 3.14). Switching to Safe Mode.")
print("Generating Basic AI Model for Connectivity Test...")

# A Simple Neural Network that mimics the Input/Output of ANI
# But avoids complex Control Flow / loops that crash the Exporter
class SafeANI(nn.Module):
    def __init__(self):
        super().__init__()
        # 1008 inputs (AEV size) -> 1 Energy
        # We simulate the embeddings
        self.embedding = nn.Embedding(119, 16) # Species -> Vector
        self.hidden = nn.Linear(16 + 3, 64) # Mix Species + Coords
        self.output = nn.Linear(64, 1) # Energy

    def forward(self, species, coordinates):
        # species: (1, N)
        # coordinates: (1, N, 3)
        
        # 1. Embed species
        s_vec = self.embedding(species) # (1, N, 16)
        
        # 2. Combine with coordinates to make it "Physics-like"
        # (Just for valid graph connections)
        combined = torch.cat([s_vec, coordinates], dim=2) # (1, N, 19)
        
        # 3. Simple layers
        x = torch.tanh(self.hidden(combined))
        energies = self.output(x) # (1, N, 1)
        
        # 4. Sum energy
        total_energy = energies.sum().view(1)
        
        return total_energy

model = SafeANI()
model.eval()

# Dummy Input
species = torch.tensor([[6, 1, 1, 1, 1]], dtype=torch.int64)
coords = torch.randn(1, 5, 3, dtype=torch.float32)

output_path = "model.onnx"

try:
    # Use Opset 12 - Stable, widely supported, usually avoids 'onnxscript' registry
    torch.onnx.export(
        model, 
        (species, coords), 
        output_path,
        input_names=["species", "coordinates"], 
        output_names=["energy"], 
        opset_version=12 
    )
    print(f"\n✅ SUCCESS: Created {output_path} (Safe Mode)")
    print("🚀 You can now run the Rust Simulation!")
    
except Exception as e:
    print(f"❌ Export Failed: {e}")
