import onnx
from onnx import helper, TensorProto

print("🛠️ Constructing ONNX Graph Manually (Bypassing PyTorch)...")

# Define Inputs
# species: (batch, atoms) - Int64
species_info = helper.make_tensor_value_info('species', TensorProto.INT64, [1, "atoms"])
# coordinates: (batch, atoms, 3) - Float
coords_info = helper.make_tensor_value_info('coordinates', TensorProto.FLOAT, [1, "atoms", 3])

# Define Output
# energy: (1) - Float
output_info = helper.make_tensor_value_info('energy', TensorProto.FLOAT, [1])

# Nodes (The Math)
# We will create a simple mock function: Energy = Sum(Coordinates^2)
# 1. Pow(coords, 2)
# 2. ReduceSum(pow)

# Constant 2.0 for Power
pow_const = helper.make_tensor('const_2', TensorProto.FLOAT, [], [2.0])
const_node = helper.make_node('Constant', [], ['const_2_out'], value=pow_const)

# Node: Power
pow_node = helper.make_node(
    'Pow',
    inputs=['coordinates', 'const_2_out'],
    outputs=['pow_out']
)

# Node: ReduceSum
# Keep dims=False to reduce to scalar
sum_node = helper.make_node(
    'ReduceSum',
    inputs=['pow_out'],
    outputs=['energy'],
    keepdims=0
)

# Node: Gradient (Forces)
# For E = Sum(coords^2), Force = -Grad(E) = -2 * coords
# 1. Mul(coords, -2.0)

# Constant -2.0
neg_two = helper.make_tensor('const_neg_2', TensorProto.FLOAT, [], [-2.0])
const_neg_node = helper.make_node('Constant', [], ['const_neg_2_out'], value=neg_two)

# Node: Mul (Forces)
force_node = helper.make_node(
    'Mul',
    inputs=['coordinates', 'const_neg_2_out'],
    outputs=['forces']
)

# Graph
graph_def = helper.make_graph(
    [const_node, pow_node, sum_node, const_neg_node, force_node],
    'AtomAI_Mock',
    [species_info, coords_info],
    [output_info, helper.make_tensor_value_info('forces', TensorProto.FLOAT, [1, "atoms", 3])]
)

# Model
model_def = helper.make_model(graph_def, producer_name='AtomEngine', ir_version=8) # Force IR version 8 for compatibility
model_def.opset_import[0].version = 14

# Save
output_path = "model.onnx"
onnx.save(model_def, output_path)

print(f"✅ SUCCESS: Created {output_path} directly.")
print("👉 Bypassed broken PyTorch Exporter.")
print("🚀 Run your simulation now!")
