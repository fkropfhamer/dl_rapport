import torch
torch.set_grad_enabled(False)

from fully_connected import FullyConnected
from sequential import Sequential
from criterion import MSELoss
from activation import ReLU, Sigmoid, Tanh

def lin_sep():
    model = Sequential([
        FullyConnected(2, 3),
        Tanh(),
        FullyConnected(3, 1),
        Tanh(),
    ])

    inputs = torch.tensor([[-1, -1], [-5, -2.5], [-7.5, 7.5], [10, 7.5], [-2.5, 12.5], [5, 10], [5, 5]]).unsqueeze(dim=1).float()
    targets = torch.tensor([[0], [0], [0], [1], [0], [1], [1]]).float()

    print(inputs.shape)

    epochs = 100

    criterion = MSELoss()

    learning_rate = 0.1

    for epoch in range(epochs):
        epoch_loss = 0

        print(f'Epoch {epoch + 1} of {epochs}')
        for x, y in zip (inputs, targets):
            output = model(x)

            loss = criterion(output, y)
            epoch_loss += loss

            error = criterion.backward(output, y)

            model.backward(error=error, learning_rate=learning_rate)

        print(f'loss {epoch_loss}')


    evaluate(model, inputs, targets)


def evaluate(model, inputs, targets):
    correct = 0
    total = 0

    for x, y in zip(inputs, targets):
        output = model(x)

        print(output, y)

        if torch.round(output) == y:
            correct += 1

        total += 1

    acc = correct / total
    print(f'Accuracy: {acc}')


def xor():
    inputs = torch.tensor([[0, 0], [1, 1], [0, 1], [1, 0]]).unsqueeze(dim=1).float()
    targets = torch.tensor([[0], [0], [1], [1]]).float()

    model = Sequential([
        FullyConnected(2, 3),
        Tanh(),
        FullyConnected(3, 1),
        Tanh(),
    ])

    epochs = 1000

    criterion = MSELoss()

    learning_rate = 0.1

    for epoch in range(epochs):
        epoch_loss = 0

        print(f'Epoch {epoch + 1} of {epochs}')
        for x, y in zip (inputs, targets):
            output = model(x)

            loss = criterion(output, y)
            epoch_loss += loss

            error = criterion.backward(output, y)

            model.backward(error=error, learning_rate=learning_rate)

        print(f'loss {epoch_loss}')


    evaluate(model, inputs, targets)



def main():
    lin_sep()
    # xor()
    




if __name__ == '__main__':
    main()