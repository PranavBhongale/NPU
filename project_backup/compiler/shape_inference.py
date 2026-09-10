import onnx


class ShapeInference:

    def __init__(self, model):

        self.model = model

    def infer(self):

        inferred_model = onnx.shape_inference.infer_shapes(
            self.model
        )

        return inferred_model