import torch
torch.set_grad_enabled(False)

class Sequential:
    def __init__(self, layers = []) -> None:
        self.layers = layers


    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def __call__(self, x):
        return self.predict(x)
