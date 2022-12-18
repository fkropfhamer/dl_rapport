import torch
torch.set_grad_enabled(False)

class MSELoss:
    def __init__(self):
        pass

    def forward(self, input, target):
        self.loss = torch.pow(torch.sub(input, target), 2)

        return self.loss

    def __call__(self, input, target):
        return self.forward(input, target)