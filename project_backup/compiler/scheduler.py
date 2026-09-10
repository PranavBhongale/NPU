class Scheduler:

    def __init__(self):

        self.schedule = []

    def add_operation(self, op):

        self.schedule.append(op)

    def print_schedule(self):

        print("Execution Schedule")

        for op in self.schedule:

            print(op)

