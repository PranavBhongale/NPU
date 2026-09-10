import onnx
class ONNXParser:

    def __init__(self, model_path):

        self.model_path = model_path
        self.model = None
        self.graph = None

    def load(self):

        self.model = onnx.load(self.model_path)
        self.graph = self.model.graph

        print("ONNX model loaded")

    def parse_nodes(self):

        nodes = []

        for node in self.graph.node:

            node_info = {

                "op_type": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "attributes": [
                    attr.name for attr in node.attribute
                ]
            }

            nodes.append(node_info)

        return nodes
    
