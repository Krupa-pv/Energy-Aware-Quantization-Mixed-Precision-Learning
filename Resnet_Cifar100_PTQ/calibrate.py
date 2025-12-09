import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as datasets
from quant_utils import quantize_model

@torch.no_grad()
def calibration_forward(model, loader, device):
    model.eval()
    for x, _ in loader:
        x = x.to(device)
        _ = model(x)

def calibrate_resnet(model, bitwidth, device):
    """
    Apply quantization hooks + run calibration.
    """
    model = quantize_model(model, bitwidth)

    transform = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408),
                    (0.2675, 0.2565, 0.2761)),
    ])

    calib_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
    calib_subset = torch.utils.data.Subset(calib_dataset, range(512))
    calib_loader = DataLoader(calib_subset, batch_size=32, shuffle=False)

    calibration_forward(model, calib_loader, device)
    print(f"Calibration complete for {bitwidth}-bit")

    return model
