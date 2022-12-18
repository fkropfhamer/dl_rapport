import torch
torch.set_grad_enabled(False)

class Sequential:
    def __init__(self, layers = []) -> None:
        self.layers = layers


    def predict(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def backward(self, error, learning_rate):
        for layer in reversed(self.layers):
            error = layer.backward(error, learning_rate)

    def __call__(self, x):
        return self.predict(x)
