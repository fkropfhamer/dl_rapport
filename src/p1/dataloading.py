from dlc_practical_prologue import generate_pair_sets
from torch.utils.data import TensorDataset, DataLoader


def load_data(batch_size=32, size=1000):
    train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(size)

    trainset = TensorDataset(train_input, train_target, train_classes)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = TensorDataset(test_input, test_target, test_classes)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader
