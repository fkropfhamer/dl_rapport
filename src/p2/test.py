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

        if torch.round(output) == y:
            correct += 1

        total += 1

    acc = correct / total
    print(f'Accuracy: {acc}')

    return acc


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





def test():

    def generate_data(size):
        x = torch.rand(size, 2).unsqueeze(dim=1)
        dis = torch.cdist(x, torch.tensor([[0.5, 0.5]]))

        import math

        d = 1/math.sqrt(2*math.pi)

        y = (dis < d).float()

        return x, y


    train_x, train_y = generate_data(1000)
    test_x, test_y = generate_data(1000)

    model = Sequential([
        FullyConnected(2, 8),
        ReLU(),
        FullyConnected(8, 12),
        Tanh(),
        FullyConnected(12, 1),
        Tanh(),
    ])

    # print(inputs.shape, train_x.shape)

    epochs = 100

    criterion = MSELoss()

    learning_rate = 0.1

    import random

    a = []
    l = []

    for epoch in range(epochs):
        epoch_loss = 0

        d = list(zip (train_x, train_y))
        random.shuffle(d)

        print(f'Epoch {epoch + 1} of {epochs}')
        for x, y in d:
            output = model(x)

            loss = criterion(output, y)
            epoch_loss += loss

            error = criterion.backward(output, y)

            model.backward(error=error, learning_rate=learning_rate)

        print(f'loss {epoch_loss / len(train_x)}')

        l.append(epoch_loss / len(train_x))

        acc = evaluate(model, test_x, test_y)
        a.append(acc)

    import matplotlib.pyplot as plt

    x = range(1, epochs + 1)


    plt.figure(0)
    plt.plot(x, l)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # plt.show()
    plt.savefig('loss.png')

    plt.figure(1)
    plt.plot(x, a)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    # plt.show()
    plt.savefig('acc.png')
    

def main():
    # lin_sep()
    # xor()

    test()
    




if __name__ == '__main__':
    main()