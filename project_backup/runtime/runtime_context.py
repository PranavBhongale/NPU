class RuntimeContext:

    def __init__(self):

        self.total_cycles = 0
        self.executed_ops = 0

        self.tensor_map = {}

    def add_cycles(self, cycles):

        self.total_cycles += cycles

    def increment_ops(self):

        self.executed_ops += 1

    def print_stats(self):

        print("Runtime Statistics")
        print("------------------")
        print(f"Executed Ops : {self.executed_ops}")
        print(f"Total Cycles : {self.total_cycles}")


