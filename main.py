from model import Net, Net2
from utils import train, test, train_loader, test_loader, optimizer, criterion, summary

def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Instantiate the model
    model = Net().to(device)

    # Display model summary
    summary(model, input_size=(1, 28, 28))

    # Set up optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1, verbose=True)

    # Set the number of epochs
    num_epochs = 20

    # Training and testing loop
    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}')
        train(model, device, train_loader, optimizer, criterion)
        test(model, device, test_loader, criterion)
        scheduler.step()

    # Plot graphs
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

if __name__ == "__main__":
    main()
 q q
