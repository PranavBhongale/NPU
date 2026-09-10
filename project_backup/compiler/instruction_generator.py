class InstructionGenerator:

    def __init__(self):

        self.instructions = []

    def generate_matmul(self):

        self.instructions.append("LOAD A")
        self.instructions.append("LOAD B")
        self.instructions.append("MATMUL")
        self.instructions.append("STORE C")

    def print_instructions(self):

        print("Generated Instructions")

        for inst in self.instructions:

            print(inst)

