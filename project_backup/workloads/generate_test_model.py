import torch
import torch.nn as nn

# -----------------------------
# GEMM MODEL
# -----------------------------

class TestGEMM(nn.Module):

    def forward(self, x, y):
        return torch.matmul(x, y)

# -----------------------------
# SIMPLE MLP
# -----------------------------

class SimpleMLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(128, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

# -----------------------------
# EXPORT FUNCTION
# -----------------------------

def export_gemm():

    model = TestGEMM()

    x = torch.randn(4, 4)
    y = torch.randn(4, 4)

    torch.onnx.export(
        model,
        (x, y),
        "../models/test_gemm.onnx",
        input_names=["x", "y"],
        output_names=["output"],
        opset_version=11
    )

    print("test_gemm.onnx generated")


def export_mlp():

    model = SimpleMLP()

    x = torch.randn(1, 128)

    torch.onnx.export(
        model,
        x,
        "../models/simple_mlp.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=11
    )

    print("simple_mlp.onnx generated")


# -----------------------------
# MAIN
# -----------------------------

if __name__ == "__main__":

    export_gemm()
    export_mlp()


