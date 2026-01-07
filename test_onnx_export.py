import torch
import torch.nn as nn
import torch.onnx

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)

def export_test():
    model = SimpleModel()
    dummy_input = torch.randn(1, 3)
    
    print("Attempting to export Simple Model...")
    try:
        torch.onnx.export(
            model, 
            dummy_input, 
            "simple_test.onnx", 
            opset_version=14,
            input_names=['input'], 
            output_names=['output']
        )
        print("SUCCESS: Exported simple_test.onnx")
    except Exception as e:
        print(f"FAILURE: {e}")

if __name__ == "__main__":
    export_test()
