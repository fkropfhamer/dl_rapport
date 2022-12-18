import torch
torch.set_grad_enabled(False)

class Linear(object):
    
    def __init__(self, in_features, out_features, bias=True) -> None:
        super().__init__()

        self.weights = torch.rand(in_features, out_features)

        self.bias = None
        if bias:
            self.bias = torch.rand(out_features)

    def forward(self, x):
        self.last_input = x

        print(self.weights.shape)

        x = torch.mm(x, self.weights)

        if self.bias:
            x += self.bias

        return x

    def backward(self, error, learning_rate):
        input_error = torch.mm(self, error, self.weights.T)
        w_error = torch.mm(self.last_input.T, error)

        self.weights -= learning_rate * w_error
        if self.bias:
            self.bias -= learning_rate * error

        return input_error
