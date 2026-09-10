from simulator.accelerator import Accelerator


class Dispatcher:

    def __init__(self):

        self.accelerator = Accelerator()

    def dispatch(self, node):

        op_type = node.op_type

        print(f"Dispatching: {op_type}")

        if op_type == "MatMul":
            self.accelerator.matmul(node)

        elif op_type == "Gemm":
            self.accelerator.gemm(node)

        elif op_type == "Conv":
            self.accelerator.conv(node)

        elif op_type == "Relu":
            self.accelerator.relu(node)

        else:
            print(f"Unsupported operator: {op_type}")

