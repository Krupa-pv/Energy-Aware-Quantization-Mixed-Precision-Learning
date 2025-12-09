import torch
import argparse
from torchvision.models import resnet18
from calibrate import calibrate_resnet
from evaluate import evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bitwidth", type=int, default=8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load trained CIFAR-100 baseline
    model = resnet18(num_classes=100)
    model.load_state_dict(torch.load("resnet18_cifar100_trained.pth"))
    model = model.to(device)

    # Apply PTQ + Calibration
    model = calibrate_resnet(model, args.bitwidth, device)

    # Evaluate quantized model
    acc = evaluate(model, device)

    torch.save(model.state_dict(), f"resnet18_ptq_{args.bitwidth}bit.pth")
    print(f"Saved → resnet18_ptq_{args.bitwidth}bit.pth")

if __name__ == "__main__":
    main()
