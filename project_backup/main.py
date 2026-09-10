from project_backup.runtime.loader import ModelLoader
from project_backup.runtime.graph_executor import GraphExecutor


MODEL_PATH = "models/test_gemm.onnx"


loader = ModelLoader(MODEL_PATH)

graph = loader.load_model()

loader.print_graph()


executor = GraphExecutor(graph)

executor.execute()