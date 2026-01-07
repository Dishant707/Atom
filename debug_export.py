import torch
import torch.nn as nn

class Trivial(nn.Module):
    def forward(self, x):
        return x + 1.0

model = Trivial()
x = torch.randn(1, 1)

print(f"Testing Export with Torch {torch.__version__}...")
try:
    torch.onnx.export(model, x, "trivial.onnx", opset_version=14)
    print("✅ Trivial Export Success")
except Exception as e:
    print(f"❌ Trivial Export Failed: {e}")
