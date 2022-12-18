import torch
torch.set_grad_enabled(False)

from linear import Linear
from sequential import Sequential
from criterion import MSELoss

def lin_sep():
    model = Sequential([
        Linear(2, 1)
    ])

    inputs = torch.tensor([[-1, -1], [-5, -2.5], [-7.5, 7.5], [10, 7.5], [-2.5, 12.5], [5, 10], [5, 5]]).unsqueeze(dim=1).float()
    targets = torch.tensor([[0], [0], [0], [1], [0], [1], [1]]).float()

    print(inputs.shape)

    epochs = 5

    criterion = MSELoss()

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1} of {epochs}')
        for x, y in zip (inputs, targets):
            output = model(x)

            loss = criterion(output, y)

            print(output, loss)



def xor():
    model = Sequential([
        Linear(2, 1)
    ])


    inputs = torch.tensor([[0, 0], [1, 1], [0, 1], [1, 0]]).unsqueeze(dim=1).float()
    targets = torch.tensor([[0], [0], [1], [1]]).float()

    print(inputs.shape)

    epochs = 5

    criterion = MSELoss()

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1} of {epochs}')
        for x, y in zip (inputs, targets):
            output = model(x)

            loss = criterion(output, y)

            print(output, loss)




def main():
    lin_sep()
    




if __name__ == '__main__':
    main()