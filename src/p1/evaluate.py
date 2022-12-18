import torch

def evaluate_model(model, testloader):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            inputs, targets, classes = data
            
            outputs = model(inputs)
            
            predictions = torch.round(torch.sigmoid(outputs))
                    
            correct += (predictions.squeeze() == targets.squeeze()).sum()
            total += predictions.size(0)

    acc = correct / total
    print(f'Accuracy: {acc}')

    return acc