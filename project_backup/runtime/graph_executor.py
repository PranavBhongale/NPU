from project_backup.runtime.dispatcher import Dispatcher


class GraphExecutor:

    def __init__(self, graph):

        self.graph = graph
        self.dispatcher = Dispatcher()

    def execute(self):

        print("Starting graph execution")

        for node in self.graph.node:

            self.dispatcher.dispatch(node)

        print("Execution complete")
