import onnx

class ModelLoader:

    def __init__(self, model_path):

        self.model_path = model_path
        self.model = None
        self.graph = None

    def load_model(self):

        self.model = onnx.load(self.model_path)
        self.graph = self.model.graph

        print(f"Loaded model: {self.model_path}")

        return self.graph

    def print_graph(self):

        for node in self.graph.node:
            print(node.op_type)


    def print_graph_accurate(self):
        for node in self.graph.node:
            print(f"Op: {node.op_type}")
            print(f"Inputs: {node.input}")
            print(f"Outputs: {node.output}")
            if node.attribute:
                for attr in node.attribute:
                    print(f"  Attr: {attr.name}")
            print("-" * 30)




