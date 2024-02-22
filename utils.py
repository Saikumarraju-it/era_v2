import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from torchsummary import summary
import torch.nn.functional as F
import matplotlib.pyplot as plt

def get_data_loaders(batch_size=512):
    train_transforms = transforms.Compose([
        transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
        transforms.Resize((28, 28)),
        transforms.RandomRotation((-15., 15.), fill=0),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
    test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)

    kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}
    test_loader = DataLoader(test_data, **kwargs)
    train_loader = DataLoader(train_data, **kwargs)

    return train_loader, test_loader

def get_device():
    use_cuda = torch.cuda.is_available()
    return torch.device("cuda" if use_cuda else "cpu")

def initialize_model(model_class):
    device = get_device()
    model = model_class().to(device)
    summary(model, input_size=(1, 28, 28))
    return model, device

def train(model, device, train_loader, optimizer, criterion, train_losses, train_acc):
    model.train()
    pbar = tqdm(train_loader)

    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        pred = model(data)
        loss = criterion(pred, target)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

        correct += (pred.argmax(dim=1) == target).sum().item()
        processed += len(data)

        pbar.set_description(desc=f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    train_acc.append(100 * correct / processed)
    train_losses.append(train_loss / len(train_loader))

def test(model, device, test_loader, criterion, test_losses, test_acc):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()
            correct += (output.argmax(dim=1) == target).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def plot_graphs(train_losses, train_acc, test_losses, test_acc):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
    plt.show()
