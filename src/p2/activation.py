from layer import Layer
import torch
torch.set_grad_enabled(False)

class Tanh(Layer):
    def __init__(self):
        pass

    def _forward(self, x):
        return torch.tanh(x)

    def backward(self, error, _):
        return (1-torch.pow(torch.tanh(self.last_input), 2)) * error


class Sigmoid(Layer):
    def __init__(self):
        pass

    def _forward(self, x):
        return torch.sigmoid(x)


    def backward(self, error, _):
        return (self.last_input*(1.0-self.last_input)).mul(error)


class ReLU(Layer):
    def __init__(self):
        pass

    def _forward(self, x):
        return torch.relu(x)
