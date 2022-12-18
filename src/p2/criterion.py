import torch
torch.set_grad_enabled(False)

class MSELoss:
    def forward(self, input, target):
        return torch.pow(torch.sub(input, target), 2)

    def backward(self, input, target):
        return torch.div(2*torch.sub(input, target), torch.prod(torch.tensor(input.shape)))


    def __call__(self, input, target):
        return self.forward(input, target)