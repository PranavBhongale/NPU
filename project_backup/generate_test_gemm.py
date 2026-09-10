
import torch
import torch.nn as nn

class TestGEMM(nn.Module):
    def forward(self, x, y):
        return torch.matmul(x, y)

model = TestGEMM()

x = torch.randn(4, 4)
y = torch.randn(4, 4)

torch.onnx.export(
    model,
    (x, y),
    "test_gemm.onnx",
    input_names=["x", "y"],
    output_names=["output"],
    opset_version=11
)

print("test_gemm.onnx generated successfully")
