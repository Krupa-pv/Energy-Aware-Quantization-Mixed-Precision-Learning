import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet18
from torch.utils.data import DataLoader

def get_loaders(batch=128):
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408),
                    (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408),
                    (0.2675, 0.2565, 0.2761)),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform_test)

    train_loader = DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=4)
    test_loader = DataLoader(testset, batch_size=batch, shuffle=False, num_workers=4)
    return train_loader, test_loader

@torch.no_grad()
def test(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        pred = out.argmax(1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)
    return 100 * correct / total

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, test_loader = get_loaders()

    model = resnet18(weights=None, num_classes=100).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    epochs = 200
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        acc = test(model, test_loader, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} | Acc: {acc:.2f}%")

    torch.save(model.state_dict(), "resnet18_cifar100_trained.pth")
    print("Model saved: resnet18_cifar100_trained.pth")

if __name__ == "__main__":
    main()
