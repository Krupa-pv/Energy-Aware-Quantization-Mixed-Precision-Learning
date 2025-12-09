import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as datasets

@torch.no_grad()
def evaluate(model, device):
    model.eval()

    transform = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408),
                    (0.2675, 0.2565, 0.2761)),
    ])

    testset = datasets.CIFAR100(root="./data", train=False,
                                download=True, transform=transform)
    loader = DataLoader(testset, batch_size=128, shuffle=False)

    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x)
        correct += preds.argmax(1).eq(y).sum().item()
        total += y.size(0)

    acc = correct / total
    print(f"Accuracy: {acc*100:.2f}%")
    return acc
