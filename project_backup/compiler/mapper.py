class Mapper:

    def __init__(self):

        self.mapping = {

            "MatMul": "SystolicArray",
            "Gemm": "SystolicArray",
            "Conv": "ConvEngine",
            "Relu": "VectorUnit",
            "Add": "VectorUnit"
        }

    def map_operator(self, op_type):

        return self.mapping.get(
            op_type,
            "Unsupported"
        )
    
