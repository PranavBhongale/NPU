WORKLOADS = {

    "test_gemm": {

        "model_path": "../models/test_gemm.onnx",

        "inputs": {
            "x": [4, 4],
            "y": [4, 4]
        },

        "description": "Basic GEMM workload",

        "operators": [
            "MatMul"
        ]
    },

    "simple_mlp": {

        "model_path": "../models/simple_mlp.onnx",

        "inputs": {
            "input": [1, 128]
        },

        "description": "Simple MLP workload",

        "operators": [
            "Gemm",
            "Relu"
        ]
    },

    "mobilenetv2": {

        "model_path": "../models/mobilenetv2.onnx",

        "inputs": {
            "input": [1, 3, 224, 224]
        },

        "description": "MobileNetV2 CNN",

        "operators": [
            "Conv",
            "Clip",
            "Add",
            "GlobalAveragePool",
            "Gemm"
        ]
    },

    "resnet18": {

        "model_path": "../models/resnet18.onnx",

        "inputs": {
            "input": [1, 3, 224, 224]
        },

        "description": "ResNet18 CNN",

        "operators": [
            "Conv",
            "Relu",
            "Add",
            "MaxPool",
            "Gemm"
        ]
    }
}


