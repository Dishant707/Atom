
import onnx
import sys

model_path = "/Users/dishant/Downloads/ani2xr_onnx_models/ani2xr_C.onnx"
if len(sys.argv) > 1:
    model_path = sys.argv[1]

print(f"Inspecting: {model_path}")
model = onnx.load(model_path)

print("Inputs:")
for input in model.graph.input:
    print(f"  {input.name}: {input.type}")

print("\nOutputs:")
for output in model.graph.output:
    print(f"  {output.name}: {output.type}")
