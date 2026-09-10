import numpy as np


class TensorManager:

    def __init__(self):

        self.tensors = {}

    def add_tensor(self, name, tensor):

        self.tensors[name] = tensor

    def get_tensor(self, name):

        return self.tensors.get(name)

    def print_tensors(self):

        for name, tensor in self.tensors.items():

            print(name, tensor.shape)

