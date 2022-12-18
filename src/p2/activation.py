from layer import Layer
import torch
torch.set_grad_enabled(False)

class Tanh(Layer):
    def _forward(self, x):
        return torch.tanh(x)

    def backward(self, error, _):
        return (1-torch.pow(torch.tanh(self.last_input), 2)) * error


class Sigmoid(Layer):
    def _forward(self, x):
        return torch.sigmoid(x)


    def backward(self, error, _):
        return (torch.sigmoid(self.last_input)*(1.0-torch.sigmoid(self.last_input))) * error


class ReLU(Layer):
    def __init__(self):
        pass

    def _forward(self, x):
        return torch.relu(x)


    def backward(self, error, _):
        x = self.last_input
        x[x < .0] = .0
        x[x > .0] = 1.

        return x * error